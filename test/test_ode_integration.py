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
    
    def test_goal_integrated_pour(self):
        """
        Demonstrates realistic pouring physics with geometric gap model.
        
        Problem: Control liquid level (p=h) by tilting cup (r=α).
        Physics: Weir discharge through geometric gap d(h,α) = L(h)·sin(α - φ(h))
        Goal: Reach target h = 0.4
        
        Key insight: Geometry creates NATURAL feedback - no goal in ODE needed!
        As h → target, gap → 0, discharge → 0, pouring stops automatically.
        
        Giskard's job: Find α(t) trajectory that minimizes ||h - h_ref ||
        subject to realistic physics h_dot = -Q(h,α)/A_t
        """
        # 1. Setup World
        world, pris_dof, rev_dof, pris_conn = self.create_initial_world()
        
        # Get symbolic variables
        p_sym = pris_dof.variables.position
        r_sym = rev_dof.variables.position
        
        # 2. Implement realistic pouring physics (geometric gap model)
        # Based on the physical model with cup geometry and weir discharge
        
        # Cup geometry parameters
        A_cup = 0.3  # Cup height [m]
        r_cup = 0.05  # Cup radius [m]
        
        # Discharge parameters (liquid - weir overflow)
        g = 9.81  # Gravity [m/s²]
        C_w = 0.55  # Weir discharge coefficient
        b = 2 * r_cup  # Weir width (cup diameter)
        A_t = 2 * r_cup * 0.10  # Cross-sectional area for h_dot calculation
        
        # Geometry functions
        # L(h) = distance from rotation axis to liquid surface corner
        # φ(h) = angle offset due to liquid level
        def L_func(h):
            return cas.sqrt((A_cup - h)**2 + r_cup**2)
        
        def phi_func(h):
            return cas.atan2((A_cup - h), r_cup)
        
        # Gap function: vertical distance liquid can spill
        # d(h, α) = L(h) * sin(α - φ(h))
        # When α < φ: d < 0 (cup tilted but not enough to pour)
        # When α > φ: d > 0 (liquid can spill by amount d)
        L_h = L_func(p_sym)
        phi_h = phi_func(p_sym)
        gap = L_h * cas.sin(r_sym - phi_h)
        
        # Effective gap (only positive part - can't pour upward!)
        # Using max(gap, 0) via smooth approximation for differentiability
        tau_smooth = 0.001
        gap_effective = tau_smooth * cas.log(1.0 + cas.exp(gap / tau_smooth))
        
        # Weir discharge equation: Q = (2/3) * C_w * b * sqrt(2*g) * d^1.5
        Q_discharge = (2.0/3.0) * C_w * b * cas.sqrt(2.0*g) * gap_effective**1.5
        
        # Plant dynamics: h_dot = -Q / A_t + Q_refill
        # Add constant refill rate to create a continuous control problem
        # Without this, Giskard can find trivial solution (don't move)
        Q_refill = 0.0002  # m³/s continuous refill rate
        ode_physics = -Q_discharge / A_t + Q_refill
        
        # Now the system has:
        # - Refill: constantly adding liquid (disturbance)
        # - Pour: controlled by tilt angle α
        # - Equilibrium: Q_discharge(h,α) = Q_refill * A_t
        # Giskard must find α to balance refill rate and maintain h = h_ref
        
        # 3. Setup Motion Statechart
        msc = MotionStatechart()
        
        # Create ODE task (pure physics, NO goal integration)
        @dataclass(eq=False, repr=False)
        class PhysicsODETask(Task):
            target_variable: cas.FloatVariable = field(kw_only=True)
            ode_function: cas.SymbolicScalar = field(kw_only=True)
            weight: float = 1.0
            
            def build(self, context: BuildContext) -> NodeArtifacts:
                cc = ConstraintCollection()
                cc.add_ode_constraint(
                    target_variable=self.target_variable,
                    ode_function=self.ode_function,
                    weight=self.weight,
                    # NO goal_value! Pure physics.
                    name=f"{self.name}_ode"
                )
                return NodeArtifacts(constraints=cc)
        
        ode_task = PhysicsODETask(
            name="pour_physics",
            target_variable=p_sym,
            ode_function=ode_physics,
            weight=10000.0  # High weight - physics must be respected
        )
        msc.add_node(ode_task)
        
        # Add position constraint on p to guide toward goal
        # This provides the OBJECTIVE: minimize ||h - h_ref||
        # Physics (ODE) provides the CONSTRAINT: how h can evolve
        pris_target = {pris_conn: 0.4}
        goal_task = JointPositionList(
            name="goal_task",
            goal_state=pris_target,
            weight=1000.0  # Strong guidance - but ODE still dominates (10000)
        )
        msc.add_node(goal_task)
        
        # Remove bias task - let Giskard figure out r naturally
        # The goal constraint on h will drive the optimization
        
        # End condition
        end = EndMotion()
        msc.add_node(end)
        end.start_condition = ode_task.observation_variable
        
        # Start conditions
        ode_task.start_condition = cas.TrinaryTrue
        goal_task.start_condition = cas.TrinaryTrue
        
        # 4. Execution
        config = QPControllerConfig.create_default_with_50hz()
        kin_sim = Executor(world=world, controller_config=config)
        kin_sim.compile(motion_statechart=msc)
        
        # Set initial positions: slightly overfull, upright
        # Start above goal to force pouring behavior
        world.state[rev_dof.name].position = 0.0  # upright
        world.state[pris_dof.name].position = 0.6  # above goal (0.4)
        
        # Tick for 20 seconds to see full pouring dynamics
        dt = config.mpc_dt
        p_vals = []
        r_vals = []
        times = []
        t = 0.0
        
        max_time = 20.0
        while t < max_time:
            kin_sim.tick()
            
            current_p = world.state[pris_dof.name].position
            current_r = world.state[rev_dof.name].position
            
            p_vals.append(current_p)
            r_vals.append(current_r)
            times.append(t)
            t += dt
            
        
        final_p = world.state[pris_dof.name].position
        final_r = world.state[rev_dof.name].position
        
        goal_value = 0.4  # Define goal for assertions
        print(f"Final state: p={final_p:.4f}, r={final_r:.4f}")
        print(f"Goal: p={goal_value}, Error: {abs(final_p - goal_value):.4f}")
        print(f"Max tilt: r_max={max(r_vals):.4f}, Min tilt: r_min={min(r_vals):.4f}")
        print(f"Tilt range: [{min(r_vals):.4f}, {max(r_vals):.4f}]")
        print(f"p range: [{min(p_vals):.4f}, {max(p_vals):.4f}]")
        
        # Generate visualization
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot 1: State evolution
        ax1.plot(times, p_vals, label='Liquid level (p)', linewidth=2)
        ax1.plot(times, r_vals, label='Tilt angle (r)', linewidth=2)
        ax1.axhline(y=goal_value, color='r', linestyle='--', label=f'Goal (p={goal_value})')
        ax1.axhline(y=0.2, color='gray', linestyle=':', label='Threshold (r=0.2)')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Position')
        ax1.set_title('Goal-Integrated ODE: Pouring Water Control')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Decay rate vs time
        p_dot = np.diff(p_vals) / dt
        ax2.plot(times[:-1], p_dot, label='dp/dt', linewidth=2, color='purple')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Decay Rate (dp/dt)')
        ax2.set_title('ODE Dynamics')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('goal_integrated_pour.png')
        print("Plot saved to goal_integrated_pour.png")
            
if __name__ == '__main__':
    unittest.main()
