import unittest
import numpy as np
import semantic_digital_twin.spatial_types.spatial_types as cas
from dataclasses import dataclass, field
from giskardpy.executor import Executor
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.graph_node import Task, NodeArtifacts, EndMotion
from giskardpy.motion_statechart.context import BuildContext
from giskardpy.qp.constraint_collection import ConstraintCollection
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.motion_statechart.data_types import LifeCycleValues, ObservationStateValues
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.connections import PrismaticConnection, RevoluteConnection
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap, Derivatives
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName

@dataclass(eq=False, repr=False)
class ODETask(Task):
    target_variable: cas.FloatVariable = field(kw_only=True)
    ode_function: cas.SymbolicScalar = field(kw_only=True)
    weight: float = 1.0
    
    def build(self, context: BuildContext) -> NodeArtifacts:
        cc = ConstraintCollection()
        cc.add_ode_constraint(
            target_variable=self.target_variable,
            ode_function=self.ode_function,
            weight=self.weight,
            name=f"{self.name}_ode"
        )
        return NodeArtifacts(constraints=cc)

class TestODEIntegration(unittest.TestCase):
    def test_ode_integration(self):
        # 1. Setup World with 1 DOF and a connection to make it active
        world = World()
        with world.modify_world():
            ul = DerivativeMap()
            ul.velocity = 10.0
            ll = DerivativeMap()
            ll.velocity = -10.0
            dof = DegreeOfFreedom(
                name=PrefixedName("h"), lower_limits=ll, upper_limits=ul
            )
            world.add_degree_of_freedom(dof)
            
            root = Body(name=PrefixedName("root"))
            tip = Body(name=PrefixedName("tip"))
            
            conn = PrismaticConnection(
                parent=root,
                child=tip,
                dof_name=dof.name,
                axis=cas.Vector3.Z(),
                name=PrefixedName("conn")
            )
            world.add_connection(conn)
        
        # Set initial position
        world.state[dof.name].position = 1.0
        
        # 2. Define ODE: h_dot = -0.5 * h
        # Use the symbolic variable from the DOF
        h_sym = dof.variables.position
        ode_function = -0.5 * h_sym
        
        # 3. Setup Motion Statechart
        msc = MotionStatechart()
        
        # Task to enforce ODE
        ode_task = ODETask(
            name="decay_task",
            target_variable=h_sym,
            ode_function=ode_function,
            weight=1.0
        )
        msc.add_node(ode_task)
        
        # Keep running until h is small enough
        end = EndMotion()
        msc.add_node(end)
        
        # Start task immediately
        ode_task.start_condition = cas.TrinaryTrue
        
        # End when h < 0.1
        end.start_condition = cas.TrinaryFalse # Never end automatically for this test part
        
        # 4. Execution
        config = QPControllerConfig.create_default_with_50hz()
        kin_sim = Executor(world=world, controller_config=config)
        kin_sim.compile(motion_statechart=msc)
        
        # Tick and verify
        dt = config.mpc_dt
        current_h = world.state[dof.name].position
        self.assertAlmostEqual(current_h, 1.0)
        
        for i in range(20): # Run for 0.4 seconds
            kin_sim.tick()
            new_h = world.state[dof.name].position
            
            # Expected velocity for implicit Euler: v = f(h) / (1 - J*dt)
            # f(h) = -0.5 * h, J = -0.5
            # v = -0.5 * h / (1 + 0.5 * dt)
            expected_v = -0.5 * current_h / (1 + 0.5 * dt)
            actual_v = world.state[dof.name].velocity
            self.assertAlmostEqual(actual_v, expected_v, places=3, msg=f"Step {i}: Velocity mismatch. Expected {expected_v}, got {actual_v}")
            
            # Check position update (approximate)
            self.assertLess(new_h, current_h, msg=f"Step {i}: h did not decrease")
            
            current_h = new_h
            
        # Final check
        self.assertLess(current_h, 1.0)
        self.assertGreater(current_h, 0.0)
        
        # Theoretical value after 0.4s: 1.0 * exp(-0.5 * 0.4) = exp(-0.2) approx 0.8187
        # Allow some error due to integration scheme
        print(current_h)
        self.assertAlmostEqual(current_h, np.exp(-0.5 * 20 * dt), delta=0.05)

    def create_initial_world(self):
        # 1. Setup World with 2 DOFs
        world = World()
        with world.modify_world():
            # Revolute DOF
            ul_rev = DerivativeMap()
            ul_rev.velocity = 0.2
            ll_rev = DerivativeMap()
            ll_rev.velocity = -0.2
            rev_dof = DegreeOfFreedom(
                name=PrefixedName("rev_joint"), lower_limits=ll_rev, upper_limits=ul_rev
            )
            world.add_degree_of_freedom(rev_dof)

            # Prismatic DOF
            ul_pris = DerivativeMap()
            ul_pris.velocity = 0.2
            ll_pris = DerivativeMap()
            ll_pris.velocity = -0.2
            pris_dof = DegreeOfFreedom(
                name=PrefixedName("pris_joint"), lower_limits=ll_pris, upper_limits=ul_pris
            )
            world.add_degree_of_freedom(pris_dof)

            ground = Body(name=PrefixedName("ground"))
            root = Body(name=PrefixedName("root"))
            child = Body(name=PrefixedName("child"))

            # Revolute Connection (World -> Root)
            rev_conn = RevoluteConnection(
                parent=ground,
                child=root,
                dof_name=rev_dof.name,
                axis=cas.Vector3.Y(),
                name=PrefixedName("rev_conn")
            )
            world.add_connection(rev_conn)

            # Prismatic Connection (Root -> Child)
            pris_conn = PrismaticConnection(
                parent=root,
                child=child,
                dof_name=pris_dof.name,
                axis=cas.Vector3.Z(),
                name=PrefixedName("pris_conn")
            )
            world.add_connection(pris_conn)

        # Set initial positions
        world.state[rev_dof.name].position = 0.0
        world.state[pris_dof.name].position = 1.0
        return world, pris_dof, rev_dof, pris_conn

    def test_mixed_constraints(self):
        plot = True
        # 1. Setup World with 2 DOFs
        world, pris_dof, rev_dof, pris_conn = self.create_initial_world()
        
        # 2. Define ODE for Prismatic: p_dot = -0.5 * p * sigmoid(r - 0.5)
        # Decay only happens if revolute joint > 0.5
        p_sym = pris_dof.variables.position
        r_sym = rev_dof.variables.position
        
        # Smooth switch (sigmoid) to make it differentiable for QP
        # 1 / (1 + exp(-k*(x - threshold)))
        switch = 1.0 / (1.0 + cas.exp(-10 * (r_sym - 0.2)))
        ode_function = -0.5 * p_sym * switch
        
        # 3. Setup Motion Statechart
        msc = MotionStatechart()
        
        # ODE Task on Prismatic
        ode_task = ODETask(
            name="decay_task",
            target_variable=p_sym,
            ode_function=ode_function,
            weight=10000.0
        )
        msc.add_node(ode_task)
        
        # Joint Position Task for Prismatic
        # Target = 0.5 * initial = 0.5
        # The solver should figure out that it needs to increase r (revolute) 
        # to enable the decay of p (prismatic) towards 0.5
        pris_target = {pris_conn: 0.4}
        pos_task = JointPositionList(
            name="pos_task",
            goal_state=pris_target,
            # weight=1.0
        )
        msc.add_node(pos_task)
        
        # End condition
        end = EndMotion()
        msc.add_node(end)
        
        ode_task.start_condition = cas.TrinaryTrue
        pos_task.start_condition = cas.TrinaryTrue
        end.start_condition = pos_task.observation_variable
        
        # 4. Execution
        config = QPControllerConfig.create_default_with_50hz()
        kin_sim = Executor(world=world, controller_config=config)
        kin_sim.compile(motion_statechart=msc)
        
        # Tick and verify
        dt = config.mpc_dt
        
        # Set initial positions
        world.state[rev_dof.name].position = 0.0
        world.state[pris_dof.name].position = 1.0
        
        # Data collection
        p_vals = []
        r_vals = []
        times = []
        t = 0.0
        
        while t < 10.0:
            kin_sim.tick()
            
            current_p = world.state[pris_dof.name].position
            current_r = world.state[rev_dof.name].position
            
            p_vals.append(current_p)
            r_vals.append(current_r)
            times.append(t)
            t += dt
            
        print(f"Final state: p={p_vals[-1]:.4f}, r={r_vals[-1]:.4f}")
        self.assertAlmostEqual(current_p, 0.4, delta=0.1,
                               msg=f"Expected {0.4}, got {current_p}")
        self.assertLess(current_r, 0.2, msg=f"Expected r < 0.2, got {current_r}")


        if plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            # Plotting
            plt.figure()
            plt.plot(times, p_vals, label='Prismatic (p)')
            plt.plot(times, r_vals, label='Revolute (r)')
            plt.xlabel('Time (s)')
            plt.ylabel('Position')
            plt.title('Mixed Constraint Test: ODE + Joint Position')
            plt.legend()
            plt.grid(True)
            plt.savefig('mixed_constraints_plot.png')
            print("Plot saved to mixed_constraints_plot.png")

    def test_mixed_constraints2(self):
        # 1. Setup World with 2 DOFs
        world, pris_dof, rev_dof, pris_conn = self.create_initial_world()

        # 2. Define ODE for Prismatic: p_dot = -0.5 * p * sigmoid(r - 0.5)
        # Decay only happens if revolute joint > 0.5
        p_sym = pris_dof.variables.position
        r_sym = rev_dof.variables.position

        ode_function = -0.5 * p_sym * 0.0
        # 3. Setup Motion Statechart
        msc = MotionStatechart()

        # ODE Task on Prismatic
        ode_task = ODETask(
            name="decay_task",
            target_variable=p_sym,
            ode_function=ode_function,
            weight=10000.0
        )
        msc.add_node(ode_task)

        # Joint Position Task for Prismatic
        # Target = 0.5 * initial = 0.5
        # The solver should figure out that it needs to increase r (revolute)
        # to enable the decay of p (prismatic) towards 0.5
        pris_target = {pris_conn: 0.4}
        pos_task = JointPositionList(
            name="pos_task",
            goal_state=pris_target,
            # weight=1.0
        )
        msc.add_node(pos_task)

        # End condition
        end = EndMotion()
        msc.add_node(end)

        ode_task.start_condition = cas.TrinaryTrue
        pos_task.start_condition = cas.TrinaryTrue
        end.start_condition = pos_task.observation_variable

        # 4. Execution
        config = QPControllerConfig.create_default_with_50hz()
        kin_sim = Executor(world=world, controller_config=config)
        kin_sim.compile(motion_statechart=msc)

        # Tick and verify
        dt = config.mpc_dt

        # Set initial positions
        world.state[rev_dof.name].position = 0.0
        world.state[pris_dof.name].position = 1.0

        # Data collection
        p_vals = []
        r_vals = []
        times = []
        t = 0.0

        while t < 10.0:
            kin_sim.tick()

            current_p = world.state[pris_dof.name].position
            current_r = world.state[rev_dof.name].position

            p_vals.append(current_p)
            r_vals.append(current_r)
            times.append(t)
            t += dt

        print(f"Final state: p={p_vals[-1]:.4f}, r={r_vals[-1]:.4f}")
        self.assertAlmostEqual(current_p, 1.0, places=2,
                               msg=f"Expected {0.4}, got {current_p}")
        self.assertLess(current_r, 0.2, msg=f"Expected r < 0.2, got {current_r}")
        self.assertGreaterEqual(current_r, 0.0, msg=f"Expected r > 0.0, got {current_r}")

    def test_mixed_constraints3(self):
        """
        Hard switching behavior (e.g. pouring water).
        Strategy: "Sim-to-Real" / Smooth Model for Hard System.
        1. Controller uses Smooth Model (Sigmoid) to have gradients.
        2. Simulation (Loop) enforces Hard Physics (Threshold).
        3. Bias task initiates the action.
        """
        # 1. Setup World
        world, pris_dof, rev_dof, pris_conn = self.create_initial_world()
        
        p_sym = pris_dof.variables.position
        r_sym = rev_dof.variables.position
        
        # Controller Model: Smooth Sigmoid
        # This gives the solver a gradient to know that reducing r stops decay
        switch_soft = 1.0 / (1.0 + cas.exp(-10 * (r_sym - 0.2)))
        ode_soft = -0.5 * p_sym * switch_soft
        
        # 2. Setup Motion Statechart
        msc = MotionStatechart()
        
        # ODE Task (High weight)
        ode_task = ODETask(
            name="decay_task",
            target_variable=p_sym,
            ode_function=ode_soft,
            weight=10000.0
        )
        msc.add_node(ode_task)
        
        # Primary Task: Drive p to 0.4
        pris_target = {pris_conn: 0.4}
        pos_task = JointPositionList(
            name="pos_task",
            goal_state=pris_target,
            weight=100.0, # High weight for primary task
            max_velocity=10.0 # Prevent early throttling
        )
        msc.add_node(pos_task)
        # msc.add_node(pos_task) # Removed duplicate addition

        # Weighted Strategy:
        # 1. Pos Task (Weight 100) is always active.
        #    CRITICAL: max_velocity=10.0 to prevent early throttling of decay.
        # 2. Bias Task (Weight 1) is always active. Encourages r -> 1.0.
        # 3. Pos Task wins when p reaches target.
        
        # Bias Task: Weak guidance
        rev_target = {world.get_connection_by_name(PrefixedName("rev_conn")): 1.0}
        bias_task = JointPositionList(
            name="bias_task",
            goal_state=rev_target,
            weight=1.0,
            threshold=0.1
        )
        msc.add_node(bias_task)
        
        # End condition
        end = EndMotion()
        msc.add_node(end)
        
        # Connect PosTask to End
        end.start_condition = pos_task.observation_variable
        
        # Start Conditions
        ode_task.start_condition = cas.TrinaryTrue
        pos_task.start_condition = cas.TrinaryTrue
        bias_task.start_condition = cas.TrinaryTrue
        end.start_condition = cas.TrinaryFalse
        
        # 3. Execution
        config = QPControllerConfig.create_default_with_50hz()
        kin_sim = Executor(world=world, controller_config=config)
        kin_sim.compile(motion_statechart=msc)
        
        # 4. Tick and verify
        dt = config.mpc_dt
        
        # Set initial positions
        world.state[rev_dof.name].position = 0.0
        world.state[pris_dof.name].position = 1.0
        
        p_vals = []
        r_vals = []
        times = []
        t = 0.0
        
        while t < 10.0:
            p_prev = world.state[pris_dof.name].position
            
            kin_sim.tick()
            
            # Enforce Hard Physics Override
            # If r < 0.2, NO decay happens in reality, regardless of what QP thought
            current_r = world.state[rev_dof.name].position
            #if current_r < 0.2:
            #    # Force p to stay constant (or only allow non-decay movement? 
            #    # Here we assume p only changes due to decay)
            #    world.state[pris_dof.name].position = p_prev
            
            current_p = world.state[pris_dof.name].position
            
            if int(t/dt) % 10 == 0:
                print(f"t={t:.2f}, p={current_p:.4f}, r={current_r:.4f}")
            
            p_vals.append(current_p)
            r_vals.append(current_r)
            times.append(t)
            t += dt
            
        print(f"Final state: p={p_vals[-1]:.4f}, r={r_vals[-1]:.4f}")
        
        # Verification
        # 1. Should have tilted past threshold to start decay
        #self.assertGreater(max(r_vals), 0.2, msg="Should have tilted past threshold at some point")
        
        # 2. Should have decayed to target (approx)
        #self.assertAlmostEqual(current_p, 0.4, delta=0.1, msg="Should have decayed to target")
        
        # 3. Should have stopped decaying (r reduced or p stable)
        # Ideally r should drop back below/near threshold to stop decay if p < 0.4
        # But since target is 0.4, it just needs to reach it.
        
        # Plotting
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(times, p_vals, label='Prismatic (p)')
        plt.plot(times, r_vals, label='Revolute (r)')
        plt.axhline(y=0.2, color='r', linestyle='--', label='Threshold')
        plt.xlabel('Time (s)')
        plt.ylabel('Position')
        plt.title('Hard Switch Test: Smooth Model Control')
        plt.legend()
        plt.grid(True)
        plt.savefig('hard_switch_plot.png')
            
if __name__ == '__main__':
    unittest.main()
