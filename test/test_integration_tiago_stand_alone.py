from copy import deepcopy

import numpy as np
import pytest
import rospy
from geometry_msgs.msg import PoseStamped, Quaternion, Point, PointStamped
from std_srvs.srv import Trigger
from tf.transformations import quaternion_about_axis, quaternion_from_matrix

import giskardpy.utils.tfwrapper as tf
from giskard_msgs.msg import MoveResult
from giskardpy import identifier
from giskardpy.configs.tiago import TiagoMujoco, TiagoStandAlone
from giskardpy.goals.goal import WEIGHT_BELOW_CA, WEIGHT_ABOVE_CA
from giskardpy.utils.utils import publish_pose
from utils_for_tests import GiskardTestWrapper

dataset = [[0.22913762274280025, 0.12668560551603925, 1.2119360989114512, 0.3831037465025321, 0.4258398112983772,
            0.700523768823976, 0.4256270948162705],
           [0.4623148254066207, 0.05798606881767965, 1.2096909667115328, 0.43262374216032773, 0.22026786557460978,
            0.5502177450806925, 0.6793962011320919],
           [0.3067547880260406, 0.22651758782045628, 1.1321989865480573, 0.7961005322525854, 0.36210459989089744,
            0.0012418747569494818, 0.48487385888712503],
           [0.649179241812018, 0.45560472473614066, 1.2414717591184654, 0.1725952889168493, 0.4690749179322401,
            0.30406609675793805, 0.8110014774419119],
           [0.43201324044849976, 0.47739986566263126, 0.9899585857097606, 0.32565334412536096, 0.4200605058221422,
            0.6481535806800708, 0.5453402669526781],
           [0.6752308274162534, 0.584388284560719, 1.21243895158353, 0.3247346845020607, 0.4928808665733212,
            0.591113934947998, 0.5497273432871727],
           [0.3361026584426964, 0.4509986292298974, 1.227271632877652, 0.6074841369583277, 0.6705586931266448,
            0.33627447582995895, 0.26121550360108486],
           [0.9513653080832589, 0.5785410075908273, 1.0178635490011214, 0.013562187107246788, 0.8443655045632064,
            0.07190673615342295, 0.5307470047775605],
           [0.2771674316308125, 0.15966703013181294, 0.9581394466480855, 0.6701129851475663, 0.06934823062102638,
            0.4845734487106768, 0.5579677256356245],
           [0.20631881106340588, 0.4133170585775281, 1.2937328578009462, 0.615901173114841, 0.02704294976956536,
            0.21502077000686165, 0.7574301897133356],
           [0.4823769141348969, 0.9983433242685429, 1.2438297972238361, 0.23271544867702493, 0.12007888642818514,
            0.3112572275586449, 0.9135335348380617],
           [0.9231751807170276, 0.6407680096414561, 1.1262682126664025, 0.05606484556106186, 0.6901870136952883,
            0.5641600232379695, 0.44969110220097475],
           [0.5962154964318122, 0.43208229926138164, 0.9434698605902225, 0.595071119073148, 0.3951026533501045,
            0.23575441184619636, 0.6589416619517942],
           [0.3324462052679874, 0.9202014425151482, 0.8588980551997569, 0.5334906898436345, 0.52190151279434,
            0.44364498349493725, 0.49617096187999477],
           [0.8867161235414418, 0.37900858071393706, 1.2463677100313393, 0.1764018716758922, 0.18284261943196292,
            0.9217429500947932, 0.29298616031998487],
           [0.37788498878224897, 0.7542046040338086, 1.1474472669070326, 0.6774525776691379, 0.30177421635319623,
            0.6050984278811729, 0.2895621175664867],
           [0.3051087645715661, 0.21704285192159167, 1.2859203422575667, 0.2724618278601589, 0.6748853759553147,
            0.06597020259340536, 0.6825996000955022],
           [0.4713196161976977, 0.5936657273561714, 1.1795440687006131, 0.2777274801445038, 0.21692302582321674,
            0.7943748249346829, 0.4947529536549289],
           [0.52298649207339, 0.9931143885674774, 1.1986717370913913, 0.6586231377678546, 0.5507069108538838,
            0.3822207503229647, 0.341825626244066],
           [0.6967117567793343, 0.5983505365392572, 1.1576500936021126, 0.13733542224477133, 0.6743980624942011,
            0.3073183941927352, 0.6571770231008663],
           [0.7920914510587204, 0.0699368101159118, 0.8610241520300492, 0.0658289973195882, 0.5652479517661319,
            0.6438586206854823, 0.5114756814405536],
           [0.3121664327578144, 0.9863148167516309, 1.1792188058092885, 0.6334720121168056, 0.5604252901902778,
            0.013870565724437355, 0.5333332085913203],
           [0.7642447792120588, 0.08704222628906744, 1.264795593617139, 0.6569899990428717, 0.18489255778583963,
            0.3601342087770084, 0.6359891782897061],
           [0.37303909412137826, 0.330221094332123, 1.106169358006408, 0.5273734199174496, 0.38002820378484153,
            0.5513210316641682, 0.5229731927521695],
           [0.9535825666902512, 0.323679383831588, 0.9618589643468687, 0.4787026489813958, 0.4566903799956209,
            0.6679001065624072, 0.34086231579858617],
           [0.7706317927990156, 0.5911052439936468, 0.9919324329284755, 0.7988235291304441, 0.10575068143377375,
            0.10005582644795744, 0.5836836422906603],
           [0.3718867111503579, 0.06653725931914589, 1.2816724372700066, 0.4331867584448287, 0.4911537952382672,
            0.7537754613576775, 0.05421932852893317],
           [0.9544861295287302, 0.530393615963066, 0.9250751054599551, 0.40188703744661547, 0.5687115643447397,
            0.4606277457876708, 0.5503417533892107],
           [0.33230385121048445, 0.373565636871227, 1.1049082969889465, 0.26473334505727064, 0.5891105978874209,
            0.30536319692571334, 0.6997272879020955],
           [0.36128331371136024, 0.6827114750354848, 1.0837854280608703, 0.14709819047912928, 0.02761286243046455,
            0.6466665390208589, 0.7479452115609446],
           [0.86980475133206, 0.504291632847863, 1.1002197773852567, 0.08104851318057536, 0.8153891899720582,
            0.5495425371422472, 0.16301719927415825],
           [0.9064084954323283, 0.6251068202513208, 1.2419585784968759, 0.47615079620937845, 0.48507317635547365,
            0.6603877756962496, 0.31917458946561894],
           [0.74287746856235, 0.04178249380901633, 0.8474518962120015, 0.690522112930541, 0.11491174133310529,
            0.3490487950331736, 0.623008380315726],
           [0.7204795277738016, 0.779960552863984, 0.8663790975048169, 0.6356587648730543, 0.5096062551976198,
            0.5167488029210863, 0.26307807583138854],
           [0.7151971223149128, 0.537329782841851, 1.2453580510969344, 0.5575379596526691, 0.522484751482242,
            0.6217365875403821, 0.17206023285035846],
           [0.9740983171527053, 0.2461984367011235, 0.8730974941667322, 0.6784838906221135, 0.4572151585545438,
            0.3383874048903765, 0.4648740401074034],
           [0.21066551048751514, 0.615892242206478, 0.9707413107627259, 0.5059969366664425, 0.03677500544779738,
            0.6478369714260035, 0.5682622260119655],
           [0.21074057596953666, 0.6583522499655254, 1.2112652285288172, 0.04158157884746464, 0.9618637525130997,
            0.25076079204032664, 0.1010352368101266],
           [0.7332473988074197, 0.17084356906138598, 0.8596145141634415, 0.7849846900271996, 0.5666403841807905,
            0.20126217405249958, 0.14903438775668115],
           [0.45091056986570777, 0.9765973153705139, 0.9353101553268934, 0.5961246050633796, 0.1277165937732708,
            0.03485613934158084, 0.7919021255579399],
           [0.1350336088560492, 0.836283629213177, 0.8082120612802393, 0.6675227247876665, 0.6887214011504136,
            0.28287146569888355, 0.007744506622460615],
           [0.4411519929333405, 0.7892403010558758, 0.8710103908612532, 0.40688763742158995, 0.19099609973144965,
            0.6584334732224129, 0.6036789724192163],
           [0.694104190638902, 0.9027263725750625, 0.9205534864016844, 0.747604387550423, 0.4538634938441014,
            0.04766558066527703, 0.4825179800699909],
           [0.6502511539133494, 0.17886345825214323, 1.2661340934209238, 0.0041060181244463746, 0.9758375309878193,
            0.04628914586194836, 0.213498404458042],
           [0.5491174809879874, 0.44677226931796676, 0.8696007110358892, 0.5114499554732508, 0.26973896676639764,
            0.2061333196366123, 0.7894104682545046],
           [0.7592540994328824, 0.340368516347477, 1.066460769252474, 0.002680984419040816, 0.8943268891012961,
            0.4357877426419281, 0.10129793243692356],
           [0.13274664747469256, 0.3838443599380629, 0.9986201526330831, 0.6722655482745409, 0.33488435396415783,
            0.006614911771930587, 0.6602028059735805],
           [0.5703936982871421, 0.15980405202062398, 0.8623546918369266, 0.7509302793092014, 0.004717641430209115,
            0.6556898311809796, 0.07843662895484144],
           [0.6053992224155381, 0.1608192910077676, 0.9239812695623362, 0.4714966753044449, 0.30232120140428637,
            0.6292643399567898, 0.5388127381726137],
           [0.8951643822728222, 0.8341211226400759, 1.1899988303098783, 0.6897642269321257, 0.009900392353881987,
            0.5517643825630587, 0.4687039146527078],
           [0.2205251839923481, 0.9171574783576605, 0.8959414432105957, 0.7061683931288525, 0.646041596434518,
            0.09496524965229114, 0.2737481645246811],
           [0.7146733477733987, 0.9896882108253516, 0.9344605343350856, 0.719254818382012, 0.3932263673212835,
            0.5711159786300753, 0.043267415348935905],
           [0.2900593505321306, 0.1574632613556367, 1.1423677670332777, 0.49619590511543993, 0.669806334978891,
            0.18910310455284918, 0.5190270833177102],
           [0.8114885448139764, 0.8037350340527635, 1.2856742615009589, 0.3429483742697199, 0.4391593800887476,
            0.7031305936537738, 0.44173840645105167],
           [0.38289610041630984, 0.7908539121704731, 1.258806889572274, 0.8641165217163084, 0.27702654838430885,
            0.3032113915102634, 0.2908982303926833],
           [0.7682082377410685, 0.7541995865955513, 1.0729994538819354, 0.296408753307824, 0.4357129802056416,
            0.4566051931512303, 0.7168038416681005],
           [0.24707872066215164, 0.32773481763801515, 0.8860978684742131, 0.5121460926352595, 0.6895755736811626,
            0.1544872400754126, 0.488186030765541],
           [0.8980536242379423, 0.6928414444438871, 0.804242035194465, 0.6631638512364426, 0.18266681419113692,
            0.1176491561316796, 0.7162438254309867],
           [0.156459461729348, 0.6828500063965525, 0.8198697605872284, 0.370848244874474, 0.24134710013068578,
            0.38450353512275737, 0.8101729371005509],
           [0.7222469749211742, 0.6612973046338372, 1.1199176689507886, 0.05800042431776729, 0.8657879381388641,
            0.12073778953417862, 0.48215099619373775],
           [0.3788892709434988, 0.9351309374378616, 1.13641623865163, 0.24507691302601425, 0.5915607053280908,
            0.3491090517899004, 0.6841901114249811],
           [0.263484946825017, 0.15136035423098027, 0.8545699690456894, 0.08391003547902125, 0.5838313356386904,
            0.29887626569105313, 0.7501820147662988],
           [0.9552677187641031, 0.9820433428189832, 0.8607154147685617, 0.4660284913517555, 0.6467957321059403,
            0.34833494336488147, 0.49308771370713766],
           [0.6274583877047605, 0.9569549301083241, 0.9548243505124863, 0.5689482452891027, 0.3654457653018476,
            0.730283194036709, 0.09712745910884539],
           [0.8765383931275051, 0.4512597379840967, 0.8449887722553702, 0.5514738979594378, 0.10371165276173998,
            0.5019579503643653, 0.6581478929669663],
           [0.23316184980694765, 0.712193048539639, 1.2513163532008034, 0.5012337119723814, 0.5746798685502428,
            0.5317570444904611, 0.3684321651270873],
           [0.012741771956835235, 0.13495880130896476, 1.0669475743311083, 0.14721326759585054, 0.2203605158856145,
            0.8263652923712055, 0.49688016708923927],
           [0.635951806007197, 0.04232795161755154, 1.1444344340191452, 0.5505978620189167, 0.4274121968934839,
            0.5245201513443836, 0.4889165768518682],
           [0.6319611735382517, 0.6008327299091452, 0.8902748338342796, 0.19146055133169151, 0.5487540382580854,
            0.5790438548927093, 0.5717692514381997],
           [0.4921466567408683, 0.6173858666716917, 0.832886862289429, 0.2779794262016512, 0.42368886155821955,
            0.7392546609107165, 0.4435287290828886],
           [0.9501354743032763, 0.6958865830866146, 0.8199252128969569, 0.3410270064382669, 0.5385039543697492,
            0.7638648699634362, 0.10111643013694682],
           [0.19989986419050232, 0.23965253040969525, 0.980130552667954, 0.6214790992031833, 0.5161517071071894,
            0.4945817559581558, 0.3205308583853996],
           [0.32361709826670604, 0.7774482301807025, 1.1896174102737536, 0.33703833506416203, 0.9211688532081911,
            0.150310892046349, 0.12353032141346731],
           [0.9066624773335783, 0.5287384846447807, 1.1949180255738596, 0.6389797394884238, 0.35714538642174093,
            0.5967119922509645, 0.3287656669800795],
           [0.14367830895423805, 0.6393213466911402, 1.021265425540571, 0.6513075674849081, 0.35485565526409213,
            0.44618405042293025, 0.5007950774639979],
           [0.4470696941840715, 0.7757105067698318, 0.877251342895766, 0.4993152027027613, 0.018833551606873052,
            0.7878201343695802, 0.36009590606624237],
           [0.7296978916110306, 0.8948753511257554, 1.2108173450508262, 0.07617857571372352, 0.6749524616373035,
            0.7266192399736613, 0.10324959676469239],
           [0.15002103611570883, 0.6715491438662066, 1.2651373125438523, 0.3290408081938747, 0.056283909022600935,
            0.4754766477900748, 0.813932568174152],
           [0.7312406121929618, 0.41878525014918244, 1.184080155191671, 0.1588142736632098, 0.41169403165278623,
            0.6457889531412621, 0.6230912282988862],
           [0.9737840659979364, 0.4206340009532389, 1.2586927936398764, 0.46106122339086375, 0.3455543091026343,
            0.6851939973569136, 0.44556026947201555],
           [0.8131158948886961, 0.22659572189066002, 1.0354106644058456, 0.7169463278987194, 0.5896503291588563,
            0.3710833629623891, 0.024445653354295164],
           [0.8862099452216309, 0.48128646985018864, 0.8670549032271856, 0.5665010453250036, 0.21076515278700264,
            0.5943342556842223, 0.5304916667928264],
           [0.9613004882993708, 0.5187554058852347, 1.2392033624938719, 0.523086151843608, 0.7128366568322271,
            0.14615689692915926, 0.44371493090241937],
           [0.09633140771700277, 0.007339203966179375, 1.2567038376427182, 0.4461021372428201, 0.5126120212463825,
            0.5344896404907538, 0.5025361907655519],
           [0.5264483510502276, 0.7434215565692516, 1.0075998355844062, 0.2862569075497792, 0.6085394257980687,
            0.4577583649590892, 0.5815445205995625],
           [0.7149496502423993, 0.030097958808667835, 1.151912444115737, 0.39173560773761756, 0.7884235067725838,
            0.4430423904870808, 0.16924842046634891],
           [0.7607445814446523, 0.051720700856322854, 1.152646622486514, 0.4483288370045849, 0.20005019611421104,
            0.4567104347624168, 0.7418872904450556],
           [0.2612234436006372, 0.21089109811885787, 0.8998655180547357, 0.018343091237226963, 0.8220782962606047,
            0.43424149066334067, 0.36781127444329903],
           [0.5362881323953261, 0.5093136537438675, 1.0171821032840096, 0.7182877626564436, 0.0454678374657111,
            0.361255829037875, 0.5928655764676762],
           [0.40171615433971397, 0.5464242577928016, 0.8200631368566245, 0.6918899641523023, 0.09724481020205915,
            0.647498085168762, 0.3042662552713957],
           [0.32711882576166373, 0.22118618612895558, 0.9504133313577405, 0.5355994653107057, 0.05979801838230738,
            0.10930018457895578, 0.8352310335514608],
           [0.5036155507899173, 0.5962361843020265, 0.8646570144147456, 0.024263393091891926, 0.9080107350545417,
            0.4167394297386293, 0.03544066142866002],
           [0.4001816166682285, 0.47024635646328017, 0.8160115071894029, 0.14051777398238505, 0.520514879632386,
            0.039779661608497593, 0.8412708207222165],
           [0.976329078996424, 0.8955264371493882, 1.0316354416888194, 0.28425996043674384, 0.7179332496630131,
            0.4457616498864063, 0.45284067331828254],
           [0.19359738811350946, 0.023712082853619698, 0.8377426844652305, 0.0023899333235850727, 0.7748889045169395,
            0.3140325288563313, 0.5485663539585196],
           [0.6338332071456131, 0.6016235514963945, 1.2568635008012903, 0.5483417668330056, 0.6433108910265829,
            0.5256509719629467, 0.09572596255693232],
           [0.6412781739330095, 0.12018681587651392, 0.9772431621458719, 0.2874883917375749, 0.3627389454142028,
            0.018035309014272395, 0.8862536937721017],
           [0.05573528465198807, 0.9908590901442009, 1.116881494133791, 0.21853397785814924, 0.4252765887015511,
            0.19892720703964253, 0.855459344401166],
           [0.3288278128168344, 0.8122040967281738, 1.2592546264278304, 0.47275507172556497, 0.015539188502181093,
            0.4048887472586242, 0.7825127974172637],
           [0.014824401257385711, 0.09415742552411599, 0.9965963362924803, 0.5149759923947974, 0.45297021272629523,
            0.6980736400133462, 0.20569615153960855]]


@pytest.fixture(scope='module')
def giskard(request, ros):
    c = TiagoTestWrapper()
    request.addfinalizer(c.tear_down)
    return c


class TiagoTestWrapper(GiskardTestWrapper):
    default_pose = {
        'torso_lift_joint': 0,
        'head_1_joint': 0.0,
        'head_2_joint': 0.0,
        'arm_left_1_joint': 0.0,
        'arm_left_2_joint': 0.0,
        'arm_left_3_joint': 0.0,
        'arm_left_4_joint': 0.0,
        'arm_left_5_joint': 0.0,
        'arm_left_6_joint': 0.0,
        'arm_left_7_joint': 0.0,
        'arm_right_1_joint': 0.0,
        'arm_right_2_joint': 0.0,
        'arm_right_3_joint': 0.0,
        'arm_right_4_joint': 0.0,
        'arm_right_5_joint': 0.0,
        'arm_right_6_joint': 0.0,
        'arm_right_7_joint': 0.0,
        'gripper_right_left_finger_joint': 0,
        'gripper_right_right_finger_joint': 0,
        'gripper_left_left_finger_joint': 0,
        'gripper_left_right_finger_joint': 0,
    }

    better_pose = {
        'arm_left_1_joint': - 1.0,
        'arm_left_2_joint': 0.0,
        'arm_left_3_joint': 1.5,
        'arm_left_4_joint': 2.2,
        'arm_left_5_joint': - 1.5,
        'arm_left_6_joint': 0.5,
        'arm_left_7_joint': 0.0,
        'arm_right_1_joint': - 1.0,
        'arm_right_2_joint': 0.0,
        'arm_right_3_joint': 1.5,
        'arm_right_4_joint': 2.2,
        'arm_right_5_joint': - 1.5,
        'arm_right_6_joint': 0.5,
        'arm_right_7_joint': 0.0,
        'torso_lift_joint': 0.35,
        'gripper_right_left_finger_joint': 0.045,
        'gripper_right_right_finger_joint': 0.045,
        'gripper_left_left_finger_joint': 0.045,
        'gripper_left_right_finger_joint': 0.045,
    }

    better_pose2 = {
        'arm_left_1_joint': 0.27,
        'arm_left_2_joint': - 1.07,
        'arm_left_3_joint': 1.5,
        'arm_left_4_joint': 2.0,
        'arm_left_5_joint': - 2.0,
        'arm_left_6_joint': 1.2,
        'arm_left_7_joint': 0.5,
        'arm_right_1_joint': 0.27,
        'arm_right_2_joint': - 1.07,
        'arm_right_3_joint': 1.5,
        'arm_right_4_joint': 2.0,
        'arm_right_5_joint': - 2.0,
        'arm_right_6_joint': 1.2,
        'arm_right_7_joint': 0.5,
    }

    def __init__(self):
        super().__init__(TiagoStandAlone)

    def move_base(self, goal_pose):
        tip_link = 'base_footprint'
        root_link = self.default_root
        self.set_json_goal(constraint_type='DiffDriveBaseGoal',
                           tip_link=tip_link, root_link=root_link,
                           goal_pose=goal_pose)
        # self.allow_all_collisions()
        self.plan_and_execute()

    def open_right_gripper(self, goal: float = 0.45):
        js = {
            'gripper_right_left_finger_joint': goal,
            'gripper_right_right_finger_joint': goal,
            'gripper_left_left_finger_joint': goal,
            'gripper_left_right_finger_joint': goal,
        }
        self.set_joint_goal(js)
        self.plan_and_execute()

    def reset(self):
        self.clear_world()
        self.reset_base()

    def reset_base(self):
        pass


@pytest.fixture()
def apartment_setup(better_pose: TiagoTestWrapper) -> TiagoTestWrapper:
    object_name = 'apartment'
    apartment_pose = PoseStamped()
    apartment_pose.header.frame_id = 'map'
    apartment_pose.pose.orientation.w = 1
    better_pose.add_urdf(name=object_name,
                         urdf=rospy.get_param('apartment_description'),
                         pose=apartment_pose)
    js = {str(k): 0.0 for k in better_pose.world.groups[object_name].movable_joints}
    better_pose.set_json_goal('SetSeedConfiguration',
                              seed_configuration=js)
    base_pose = PoseStamped()
    base_pose.header.frame_id = 'side_B'
    base_pose.pose.position.x = 1.5
    base_pose.pose.position.y = 2.4
    base_pose.pose.orientation.w = 1
    better_pose.set_json_goal('SetOdometry',
                              group_name='tiago_dual',
                              base_pose=base_pose)
    better_pose.allow_all_collisions()
    better_pose.move_base(base_pose)
    return better_pose


class TestCartGoals:

    # def test_random_eef_goals(self, apartment_setup: TiagoTestWrapper):
    #     tip_link = 'arm_left_tool_link'
    #     base_pose = PoseStamped()
    #     base_pose.header.frame_id = 'side_B'
    #     base_pose.pose.position.x = 1.7
    #     base_pose.pose.position.y = 1.
    #     base_pose.pose.orientation.w = 1
    #     successes = []
    #     for i, goal in enumerate(dataset):
    #         print(i)
    #         pgoal = PoseStamped()
    #         pgoal.header.frame_id = 'iai_apartment/cabinet5'
    #         pgoal.pose.position = Point(goal[0], goal[1], goal[2])
    #         pgoal.pose.position.z -= 0.5
    #         pgoal.pose.orientation = Quaternion(goal[3], goal[4], goal[5], goal[6])
    #         publish_pose(pgoal)
    #         pgoal.header.frame_id = 'cabinet5'
    #         try:
    #             apartment_setup.set_prediction_horizon(1)
    #             apartment_setup.set_seed_configuration(apartment_setup.better_pose)
    #             apartment_setup.set_json_goal('SetOdometry',
    #                                           group_name='tiago_dual',
    #                                           base_pose=base_pose)
    #             apartment_setup.set_json_goal('KeepHandInWorkspace',
    #                                       map_frame='map',
    #                                       base_footprint='base_footprint',
    #                                       tip_link=tip_link)
    #             apartment_setup.set_cart_goal(pgoal, tip_link, root_link='map')
    #             # better_pose.allow_all_collisions()
    #             apartment_setup.plan_and_execute()
    #             successes.append(True)
    #         except Exception as e:
    #             print(e)
    #             successes.append(False)
    #     apartment_setup.god_map.get_data(identifier.timer_collector).pretty_print(lambda i, x: i != 0 and i % 2 == 0)
    #     print(f'successes: {successes}')

    def test_drive_topright_bottom_left(self, zero_pose: TiagoTestWrapper):
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position = Point(0.489, -0.598, 0.000)
        goal.pose.orientation.w = 1
        zero_pose.allow_all_collisions()
        zero_pose.move_base(goal)

        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position = Point(-0.026, 0.569, 0.000)
        goal.pose.orientation = Quaternion(0, 0, 0.916530200374776, 0.3999654882623912)
        zero_pose.allow_all_collisions()
        zero_pose.move_base(goal)

    def test_drive_forward_forward(self, zero_pose: TiagoTestWrapper):
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position.x = 1
        goal.pose.position.y = 1
        # goal.pose.orientation.w = 1
        goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 4, [0, 0, 1]))
        zero_pose.allow_all_collisions()
        zero_pose.move_base(goal)
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position.x = 2
        goal.pose.position.y = 0
        # goal.pose.orientation.w = 1
        goal.pose.orientation = Quaternion(*quaternion_about_axis(-np.pi / 4, [0, 0, 1]))
        zero_pose.allow_all_collisions()
        zero_pose.move_base(goal)
        # zero_pose.set_translation_goal(goal, 'base_footprint', 'odom')
        # zero_pose.plan_and_execute()

    def test_drive_backward_backward(self, zero_pose: TiagoTestWrapper):
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position.x = -1
        goal.pose.position.y = -1
        # goal.pose.orientation.w = 1
        goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 4, [0, 0, 1]))
        zero_pose.allow_all_collisions()
        zero_pose.move_base(goal)
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position.x = -2
        goal.pose.position.y = 0
        # goal.pose.orientation.w = 1
        goal.pose.orientation = Quaternion(*quaternion_about_axis(-np.pi / 4, [0, 0, 1]))
        zero_pose.allow_all_collisions()
        zero_pose.move_base(goal)
        # zero_pose.set_translation_goal(goal, 'base_footprint', 'odom')
        # zero_pose.plan_and_execute()

    def test_drive_left(self, zero_pose: TiagoTestWrapper):
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position.x = 0.01
        goal.pose.position.y = 0.5
        goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 8, [0, 0, 1]))
        zero_pose.allow_all_collisions()
        zero_pose.move_base(goal)

    def test_drive_left_right(self, zero_pose: TiagoTestWrapper):
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position.x = 0.01
        goal.pose.position.y = 0.5
        goal.pose.orientation = Quaternion(*quaternion_about_axis(-np.pi, [0, 0, 1]))
        zero_pose.allow_all_collisions()
        zero_pose.move_base(goal)
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position.x = -0.01
        goal.pose.position.y = -0.5
        goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi * 0.2, [0, 0, 1]))
        zero_pose.allow_all_collisions()
        zero_pose.move_base(goal)

    def test_drive7(self, apartment_setup: TiagoTestWrapper):
        apartment_setup.set_joint_goal(apartment_setup.better_pose2)
        apartment_setup.plan_and_execute()

        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position.x = 1.1
        goal.pose.position.y = 2
        goal.pose.orientation = Quaternion(*quaternion_about_axis(-np.pi, [0, 0, 1]))
        apartment_setup.set_joint_goal(apartment_setup.better_pose2)
        apartment_setup.move_base(goal)

        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position.x = 1.1
        goal.pose.position.y = 3
        goal.pose.orientation = Quaternion(*quaternion_about_axis(-np.pi, [0, 0, 1]))
        # apartment_setup.set_joint_goal(apartment_setup.better_pose2)
        apartment_setup.move_base(goal)

    def test_drive3(self, apartment_setup: TiagoTestWrapper):
        countertop_frame = 'iai_apartment/island_countertop'

        start_base_pose = PoseStamped()
        start_base_pose.header.frame_id = 'map'
        start_base_pose.pose.position = Point(1.295, 2.294, 0)
        start_base_pose.pose.orientation = Quaternion(0.000, 0.000, 0.990, -0.139)
        # apartment_setup.allow_all_collisions()
        apartment_setup.move_base(start_base_pose)

        countertip_P_goal = PointStamped()
        countertip_P_goal.header.frame_id = countertop_frame
        countertip_P_goal.point.x = 1.3
        countertip_P_goal.point.y = -0.3
        map_P_goal = tf.msg_to_homogeneous_matrix(tf.transform_point('map', countertip_P_goal))
        map_P_goal[-2] = 0

        # map_P_base_footprint = tf.msg_to_homogeneous_matrix(tf.lookup_point('map', 'base_footprint'))
        # # map_P_goal = np.array([1.3, -0.3, 0, 1])
        # x = map_P_goal - map_P_base_footprint
        # x = x[:3]
        # x /= np.linalg.norm(x)
        # z = np.array([0,0,1])
        # y = np.cross(z, x)
        # y /= np.linalg.norm(y)
        # map_R_goal = np.vstack([x,y,z]).T
        # map_R_goal = np.vstack([np.vstack([map_R_goal.T, [0,0,0]]).T, [0,0,0,1]])
        #
        # base_pose = tf.lookup_pose('map', 'base_footprint')
        # base_pose.pose.orientation = Quaternion(*quaternion_from_matrix(map_R_goal))
        # # base_pose = tf.transform_pose(apartment_setup.default_root, base_pose)
        # # base_pose.pose.position.z = 0
        # apartment_setup.allow_all_collisions()
        # apartment_setup.move_base(base_pose)

        base_pose = PoseStamped()
        base_pose.header.frame_id = countertop_frame
        base_pose.pose.position.x = 1.3
        base_pose.pose.position.y = -0.3
        base_pose.pose.orientation = Quaternion(*quaternion_from_matrix([[-1, 0, 0, 0],
                                                                         [0, -1, 0, 0],
                                                                         [0, 0, 1, 0],
                                                                         [0, 0, 0, 1]]))
        base_pose = tf.transform_pose(apartment_setup.default_root, base_pose)
        base_pose.pose.position.z = 0
        # apartment_setup.allow_all_collisions()
        apartment_setup.move_base(base_pose)


class TestCollisionAvoidance:
    def test_grasp_cup(self, apartment_setup: TiagoTestWrapper):
        base_pose = PoseStamped()
        base_pose.header.frame_id = 'map'
        base_pose.pose.position = Point(1.271, 2.021, 0.000)
        base_pose.pose.orientation = Quaternion(0.000, 0.000, 0.969, -0.247)
        start_js = {
            'arm_left_1_joint': 0.2991742346702665,
            'arm_left_2_joint': -1.0601210227400328,
            'arm_left_3_joint': 1.5292663431621898,
            'arm_left_4_joint': 1.989600908762828,
            'arm_left_5_joint': -1.9884850218240293,
            'arm_left_6_joint': 1.2035198112338543,
            'arm_left_7_joint': 0.5389288929611318,
            'arm_right_1_joint': 0.26824576344275874,
            'arm_right_2_joint': -1.0981129751261716,
            'arm_right_3_joint': 1.4993808806754207,
            'arm_right_4_joint': 2.0443788079372287,
            'arm_right_5_joint': -2.011493907931497,
            'arm_right_6_joint': 1.200471509340981,
            'arm_right_7_joint': 0.5016044145676323,
            'gripper_left_left_finger_joint': 0.03999283337664056,
            'gripper_left_right_finger_joint': 0.039602112240903674,
            'gripper_right_left_finger_joint': 0.04368562830567374,
            'gripper_right_right_finger_joint': 0.04378708138690458,
            'head_1_joint': -0.04947071397908502,
            'head_2_joint': 0.006342790951699928,
            'torso_lift_joint': 0.34363201970443347,
        }
        apartment_setup.set_seed_configuration(start_js)
        grasp_goal = PoseStamped()
        grasp_goal.header.frame_id = 'map'
        grasp_goal.pose.position = Point(0.7204, 1.578, 1.4419)
        grasp_goal.pose.orientation = Quaternion(0.001145436, 0.7069, 0.707274, -3.63593e-4)
        apartment_setup.set_cart_goal(grasp_goal, tip_link='arm_left_tool_link',
                                      root_link='odom')
        # apartment_setup.set_cart_goal(grasp_goal, tip_link='arm_right_tool_link',
        #                               root_link='odom')
        apartment_setup.plan_and_execute()
        # <3D-VECTOR (0.7204197676719738d0 1.5780217069225657d0 1.4419153928756714d0)>
        # <QUATERNION (-0.0011454361739371421d0 0.7069380921005101d0 0.7072744425321389d0 -3.635934990000589d-4)>>

    def test_avoid_all(self, zero_pose: TiagoTestWrapper):
        zero_pose.set_joint_goal(zero_pose.better_pose)
        zero_pose.avoid_collision(0.1, group1=zero_pose.get_robot_name())
        zero_pose.plan_and_execute()

    def test_attach_object(self, better_pose):
        parent_link = 'arm_left_tool_link'
        box_name = 'box'
        box_pose = PoseStamped()
        box_pose.header.frame_id = parent_link
        box_pose.pose.position.x = 0.2
        box_pose.pose.orientation.w = 1
        better_pose.add_box(box_name,
                            (0.1, 0.1, 0.1),
                            pose=box_pose,
                            parent_link=parent_link,
                            parent_link_group=better_pose.get_robot_name())
        better_pose.set_joint_goal(better_pose.default_pose)
        better_pose.plan_and_execute()

    def test_demo1(self, apartment_setup: TiagoTestWrapper):
        # setup
        apartment_name = 'apartment'
        l_tcp = 'gripper_left_grasping_frame'
        r_tcp = 'gripper_right_grasping_frame'
        handle_name = 'handle_cab1_top_door'
        handle_name_frame = 'handle_cab1_top_door'
        cupboard_floor_frame = 'cabinet1_coloksu_level4'
        cupboard_floor = 'cabinet1_coloksu_level4'
        base_footprint = 'base_footprint'
        countertop_frame = 'island_countertop'
        countertop = 'island_countertop'
        grasp_offset = 0.1
        start_base_pose = apartment_setup.world.compute_fk_pose(apartment_setup.default_root, base_footprint)

        # spawn cup
        cup_name = 'cup'
        cup_pose = PoseStamped()
        cup_pose.header.frame_id = cupboard_floor_frame
        cup_height = 0.1653
        cup_pose.pose.position.z = cup_height / 2
        cup_pose.pose.position.x = 0.15
        cup_pose.pose.position.y = 0.15
        cup_pose.pose.orientation.w = 1
        apartment_setup.add_box(name=cup_name,
                                size=(0.0753, 0.0753, cup_height),
                                pose=cup_pose,
                                parent_link=cupboard_floor,
                                parent_link_group=apartment_name)

        # open cupboard
        goal_angle = np.pi / 2
        left_pose = PoseStamped()
        left_pose.header.frame_id = handle_name_frame
        left_pose.pose.position.x = -grasp_offset
        left_pose.pose.orientation = Quaternion(*quaternion_from_matrix([[1, 0, 0, 0],
                                                                         [0, -1, 0, 0],
                                                                         [0, 0, -1, 0],
                                                                         [0, 0, 0, 1]]))
        apartment_setup.set_cart_goal(left_pose,
                                      tip_link=l_tcp,
                                      root_link=apartment_setup.default_root,
                                      weight=WEIGHT_ABOVE_CA * 10,
                                      check=False)
        goal_point = PointStamped()
        goal_point.header.frame_id = 'cabinet1_door_top_left'
        # apartment_setup.set_diff_drive_tangential_to_point(goal_point)
        apartment_setup.set_keep_hand_in_workspace(tip_link=l_tcp)
        # apartment_setup.avoid_joint_limits(50)
        apartment_setup.plan_and_execute()

        apartment_setup.set_diff_drive_tangential_to_point(goal_point=goal_point, weight=WEIGHT_BELOW_CA)
        apartment_setup.set_cart_goal(left_pose,
                                      tip_link=l_tcp,
                                      root_link=apartment_setup.default_root,
                                      check=False)
        apartment_setup.plan_and_execute()

        apartment_setup.set_json_goal('Open',
                                      tip_link=l_tcp,
                                      environment_link=handle_name,
                                      goal_joint_state=goal_angle)
        apartment_setup.set_diff_drive_tangential_to_point(goal_point=goal_point)
        apartment_setup.set_avoid_joint_limits_goal(
            joint_list=['arm_left_1_joint', 'arm_left_2_joint', 'arm_left_3_joint',
                        'arm_left_4_joint', 'arm_left_5_joint', 'arm_left_6_joint',
                        'arm_left_7_joint'])
        apartment_setup.plan_and_execute()

        # grasp cup
        apartment_setup.set_joint_goal(apartment_setup.better_pose, check=False)
        apartment_setup.plan_and_execute()
        apartment_setup.move_base(start_base_pose)

        grasp_pose = PoseStamped()
        grasp_pose.header.frame_id = cup_name
        grasp_pose.pose.position.x = grasp_offset
        grasp_pose.pose.orientation = Quaternion(*quaternion_from_matrix([[-1, 0, 0, 0],
                                                                          [0, -1, 0, 0],
                                                                          [0, 0, 1, 0],
                                                                          [0, 0, 0, 1]]))
        apartment_setup.set_cart_goal(goal_pose=grasp_pose,
                                      tip_link=l_tcp,
                                      root_link=apartment_setup.default_root)
        apartment_setup.set_keep_hand_in_workspace(tip_link=l_tcp)
        apartment_setup.plan_and_execute()
        apartment_setup.update_parent_link_of_group(cup_name, l_tcp)

        # place cup
        apartment_setup.set_joint_goal(apartment_setup.better_pose, check=False)
        apartment_setup.plan_and_execute()
        base_pose = PoseStamped()
        base_pose.header.frame_id = countertop_frame
        base_pose.pose.position.x = 1.3
        base_pose.pose.position.y = -0.3
        base_pose.pose.orientation = Quaternion(*quaternion_from_matrix([[-1, 0, 0, 0],
                                                                         [0, -1, 0, 0],
                                                                         [0, 0, 1, 0],
                                                                         [0, 0, 0, 1]]))
        base_pose = apartment_setup.transform_msg(apartment_setup.default_root, base_pose)
        base_pose.pose.position.z = 0
        apartment_setup.move_base(base_pose)

        cup_place_pose = PoseStamped()
        cup_place_pose.header.frame_id = countertop_frame
        cup_place_pose.pose.position.x = 0.25
        # cup_place_pose.pose.position.y = 0.
        cup_place_pose.pose.position.z = 0.02
        cup_place_pose.pose.orientation = Quaternion(*quaternion_from_matrix([[-1, 0, 0, 0],
                                                                              [0, -1, 0, 0],
                                                                              [0, 0, 1, 0],
                                                                              [0, 0, 0, 1]]))
        apartment_setup.set_cart_goal(goal_pose=cup_place_pose,
                                      tip_link=cup_name,
                                      root_link=apartment_setup.default_root)
        apartment_setup.plan_and_execute()

    def test_close_cabinet(self, apartment_setup: TiagoTestWrapper):
        # setup
        apartment_name = 'apartment'
        l_tcp = 'gripper_left_grasping_frame'
        r_tcp = 'gripper_right_grasping_frame'
        handle_name = 'handle_cab1_top_door'
        handle_name_frame = 'handle_cab1_top_door'
        cupboard_floor_frame = 'cabinet1_coloksu_level4'
        cupboard_floor = 'cabinet1_coloksu_level4'
        base_footprint = 'base_footprint'
        countertop_frame = 'island_countertop'
        countertop = 'island_countertop'
        grasp_offset = 0.1
        goal_angle = np.pi / 2
        start_base_pose = apartment_setup.world.compute_fk_pose(apartment_setup.default_root, base_footprint)
        start_base_pose.pose.position.y -= 1
        start_base_pose.pose.position.x += 0.3
        start_base_pose.pose.orientation = Quaternion(*quaternion_about_axis(np.pi*3/4, [0,0,1]))
        apartment_setup.set_seed_odometry(start_base_pose)

        apartment_state = {
            'cabinet1_door_top_left_joint': goal_angle
        }
        apartment_setup.set_seed_configuration(apartment_state)
        apartment_setup.set_seed_configuration(apartment_setup.better_pose2)

        left_pose = PoseStamped()
        left_pose.header.frame_id = handle_name_frame
        left_pose.pose.position.x = -grasp_offset
        left_pose.pose.orientation = Quaternion(*quaternion_from_matrix([[1, 0, 0, 0],
                                                                         [0, -1, 0, 0],
                                                                         [0, 0, -1, 0],
                                                                         [0, 0, 0, 1]]))
        apartment_setup.set_cart_goal(left_pose,
                                      tip_link=l_tcp,
                                      root_link=apartment_setup.default_root,
                                      weight=WEIGHT_ABOVE_CA * 10,
                                      check=False)
        goal_point = PointStamped()
        goal_point.header.frame_id = 'cabinet1_door_top_left'
        goal_point = apartment_setup.transform_msg('map', goal_point)
        # apartment_setup.set_diff_drive_tangential_to_point(goal_point)
        apartment_setup.set_keep_hand_in_workspace(tip_link=l_tcp)
        # apartment_setup.avoid_joint_limits(50)
        # apartment_setup.set_diff_drive_tangential_to_point(goal_point=goal_point)
        apartment_setup.plan_and_execute()

        pointing_point = PointStamped()
        pointing_point.header.frame_id = 'gripper_left_grasping_frame'
        apartment_setup.set_pointing_goal(tip_link='xtion_optical_frame',
                                          root_link='gripper_left_grasping_frame',
                                          goal_point=pointing_point,
                                          check=False)

        apartment_setup.set_cart_goal(left_pose,
                                      tip_link=l_tcp,
                                      root_link=apartment_setup.default_root,
                                      weight=WEIGHT_ABOVE_CA * 10,
                                      check=False)
        # apartment_setup.set_diff_drive_tangential_to_point(goal_point)
        # apartment_setup.set_keep_hand_in_workspace(tip_link=l_tcp)
        # apartment_setup.avoid_joint_limits(50)
        apartment_setup.set_diff_drive_tangential_to_point(goal_point=goal_point)
        apartment_setup.plan_and_execute()
        # type: "DiffDriveTangentialToPoint"
        # parameter_value_pair: "{\"goal_point\":{\"message_type\":\"geometry_msgs/PointStamped\",\"message\":{\"\
    # header\":{\"stamp\":{\"secs\":0,\"nsecs\":0.0},\"frame_id\":\"map\",\"seq\":0},\"\
    # point\":{\"x\":0.563000037094409,\"y\":1.1559999675637478,\"z\":1.39499994913737}}}}"
        apartment_setup.set_json_goal('Close',
                                      tip_link=l_tcp,
                                      environment_link=handle_name)
        apartment_setup.set_diff_drive_tangential_to_point(goal_point=goal_point)
        apartment_setup.set_avoid_joint_limits_goal(
            joint_list=['arm_left_1_joint', 'arm_left_2_joint', 'arm_left_3_joint', 'arm_left_4_joint',
                        'arm_left_5_joint', 'arm_left_6_joint', 'arm_left_7_joint'])
        apartment_setup.plan_and_execute()



    def test_self_collision_avoidance(self, zero_pose: TiagoTestWrapper):
        js = {
            'arm_left_1_joint': -1.1069832458862692,
            'arm_left_2_joint': 1.4746164329656843,
            'arm_left_3_joint': 2.7736173839819602,
            'arm_left_4_joint': 1.6237723180496708,
            'arm_left_5_joint': -1.5975088318771629,
            'arm_left_6_joint': 1.3300843607103001,
            'arm_left_7_joint': -0.016546381784501657,
            'arm_right_1_joint': -1.0919070230703032,
            'arm_right_2_joint': 1.4928456221831905,
            'arm_right_3_joint': 2.740050318770805,
            'arm_right_4_joint': 1.6576417817518292,
            'arm_right_5_joint': -1.4619211253492215,
            'arm_right_6_joint': 1.2787860569647924,
            'arm_right_7_joint': 0.013613188642612156,
            'gripper_left_left_finger_joint': 0.0393669359310417,
            'gripper_left_right_finger_joint': 0.04396903656716549,
            'gripper_right_left_finger_joint': 0.03097991016001716,
            'gripper_right_right_finger_joint': 0.04384773311365822,
            'head_1_joint': -0.10322685494051058,
            'head_2_joint': -1.0027367693813412,
            'torso_lift_joint': 0.2499968644929236,
        }
        # zero_pose.set_joint_goal(js)
        # zero_pose.allow_all_collisions()
        # zero_pose.plan_and_execute()
        zero_pose.set_seed_configuration(js)
        # zero_pose.set_joint_goal(zero_pose.better_pose2)
        js2 = {
            'torso_lift_joint': 0.3400000002235174,
        }
        zero_pose.set_joint_goal(js2)
        zero_pose.plan()

    def test_avoid_self_collision4(self, zero_pose: TiagoTestWrapper):
        js = {
            'arm_left_1_joint': 0.21181287002285662,
            'arm_left_2_joint': -0.6151379734525764,
            'arm_left_3_joint': 0.769352860826213,
            'arm_left_4_joint': 1.5366410535725352,
            'arm_left_5_joint': 0.6692852960138725,
            'arm_left_6_joint': 0.8499649769704987,
            'arm_left_7_joint': 0.3934248653346525,
            'arm_right_1_joint': 0.2605757577669546,
            'arm_right_2_joint': -1.1145267872723925,
            'arm_right_3_joint': 1.4016496683543236,
            'arm_right_4_joint': 2.1447945222448572,
            'arm_right_5_joint': -2.0050615624226524,
            'arm_right_6_joint': 1.2321070888078671,
            'arm_right_7_joint': 0.5130944511015763,
            'gripper_left_left_finger_joint': 0.0016253199614312491,
            'gripper_left_right_finger_joint': 0.0015128278396200111,
            'gripper_right_left_finger_joint': 0.043748218050233635,
            'gripper_right_right_finger_joint': 0.04384773311365822,
            'head_1_joint': 1.2314613306445925,
            'head_2_joint': 0.4333201391595926,
        }

        zero_pose.set_joint_goal(js, check=False)
        zero_pose.plan_and_execute()
        zero_pose.set_joint_goal(zero_pose.better_pose2)
        zero_pose.plan_and_execute()

    def test_left_arm(self, zero_pose: TiagoTestWrapper):
        box_pose = PoseStamped()
        box_pose.header.frame_id = 'arm_left_3_link'
        box_pose.pose.position.z = 0.07
        box_pose.pose.position.x = 0.1
        box_pose.pose.orientation.w = 1
        # zero_pose.add_box('box',
        #                   size=(0.05,0.05,0.05),
        #                   pose=box_pose)
        box_pose = PoseStamped()
        box_pose.header.frame_id = 'arm_left_5_link'
        box_pose.pose.position.z = 0.07
        box_pose.pose.position.y = -0.1
        box_pose.pose.orientation.w = 1
        # zero_pose.add_box('box2',
        #                   size=(0.05,0.05,0.05),
        #                   pose=box_pose)
        # zero_pose.allow_self_collision()
        zero_pose.plan_and_execute()

    def test_load_negative_scale(self, zero_pose: TiagoTestWrapper):
        mesh_path = 'package://tiago_description/meshes/arm/arm_3_collision.dae'
        box_pose = PoseStamped()
        box_pose.header.frame_id = 'base_link'
        box_pose.pose.position.x = 0.6
        box_pose.pose.position.z = 0.0
        box_pose.pose.orientation.w = 1
        zero_pose.add_mesh('meshy',
                           mesh=mesh_path,
                           pose=box_pose,
                           scale=(1, 1, -1),
                           )
        box_pose = PoseStamped()
        box_pose.header.frame_id = 'base_link'
        box_pose.pose.position.x = 0.6
        box_pose.pose.position.z = -0.1
        box_pose.pose.orientation.w = 1
        zero_pose.add_box('box1',
                          size=(0.1, 0.1, 0.01),
                          pose=box_pose,
                          parent_link='base_link',
                          parent_link_group=zero_pose.get_robot_name())
        box_pose = PoseStamped()
        box_pose.header.frame_id = 'base_link'
        box_pose.pose.position.x = 0.6
        box_pose.pose.position.y = 0.1
        box_pose.pose.position.z = 0.05
        box_pose.pose.orientation.w = 1
        zero_pose.add_box('box2',
                          size=(0.1, 0.01, 0.1),
                          pose=box_pose,
                          parent_link='base_link',
                          parent_link_group=zero_pose.get_robot_name())
        box_pose = PoseStamped()
        box_pose.header.frame_id = 'base_link'
        box_pose.pose.position.x = 0.6
        box_pose.pose.position.y = -0.1
        box_pose.pose.position.z = 0.05
        box_pose.pose.orientation.w = 1
        zero_pose.add_box('box3',
                          size=(0.1, 0.01, 0.1),
                          pose=box_pose,
                          parent_link='base_link',
                          parent_link_group=zero_pose.get_robot_name())
        # box_pose = PoseStamped()
        # box_pose.header.frame_id = 'base_link'
        # box_pose.pose.position.x = 0.6
        # box_pose.pose.orientation.w = 1
        # zero_pose.add_mesh('meshy2',
        #                    mesh=mesh_path,
        #                    pose=box_pose,
        #                    scale=(1, 1, 1),
        #                    )
        zero_pose.plan_and_execute()

    def test_drive_into_kitchen(self, apartment_setup: TiagoTestWrapper):
        base_goal = PoseStamped()
        base_goal.header.frame_id = 'base_footprint'
        base_goal.pose.position.x = 2
        base_goal.pose.orientation.w = 1
        apartment_setup.move_base(base_goal)

    def test_open_cabinet_left(self, apartment_setup: TiagoTestWrapper):
        tcp = 'gripper_left_grasping_frame'
        handle_name = 'handle_cab1_top_door'
        handle_name_frame = 'handle_cab1_top_door'
        goal_angle = np.pi / 2
        left_pose = PoseStamped()
        left_pose.header.frame_id = handle_name_frame
        left_pose.pose.position.x = -0.1
        left_pose.pose.orientation = Quaternion(*quaternion_from_matrix([[1, 0, 0, 0],
                                                                         [0, -1, 0, 0],
                                                                         [0, 0, -1, 0],
                                                                         [0, 0, 0, 1]]))
        apartment_setup.set_cart_goal(left_pose,
                                      tip_link=tcp,
                                      root_link=apartment_setup.world.root_link_name,
                                      check=False)
        goal_point = PointStamped()
        goal_point.header.frame_id = 'cabinet1_door_top_left'
        apartment_setup.set_json_goal('DiffDriveTangentialToPoint',
                                      goal_point=goal_point)
        apartment_setup.set_avoid_joint_limits_goal(50)
        apartment_setup.plan_and_execute()

        apartment_setup.set_json_goal('Open',
                                      tip_link=tcp,
                                      environment_link=handle_name,
                                      goal_joint_state=goal_angle)
        apartment_setup.set_json_goal('DiffDriveTangentialToPoint',
                                      goal_point=goal_point)
        apartment_setup.set_avoid_joint_limits_goal(50)
        apartment_setup.plan_and_execute()

        apartment_setup.set_json_goal('Open',
                                      tip_link=tcp,
                                      environment_link=handle_name,
                                      goal_joint_state=0)
        apartment_setup.set_json_goal('DiffDriveTangentialToPoint',
                                      goal_point=goal_point)
        apartment_setup.set_avoid_joint_limits_goal(50)
        apartment_setup.plan_and_execute()

    def test_open_cabinet_right(self, apartment_setup: TiagoTestWrapper):
        base_pose = PoseStamped()
        base_pose.header.frame_id = 'iai_apartment/side_B'
        base_pose.pose.position.x = 1.6
        base_pose.pose.position.y = 3.1
        base_pose.pose.orientation.w = 1
        base_pose = tf.transform_pose(tf.get_tf_root(), base_pose)
        apartment_setup.set_localization(base_pose)
        tcp = 'gripper_right_grasping_frame'
        handle_name = 'handle_cab1_top_door'
        handle_name_frame = 'iai_apartment/handle_cab1_top_door'
        goal_angle = np.pi / 2
        left_pose = PoseStamped()
        left_pose.header.frame_id = handle_name_frame
        left_pose.pose.position.x = -0.1
        left_pose.pose.orientation.w = 1
        apartment_setup.set_cart_goal(left_pose,
                                      tip_link=tcp,
                                      root_link=tf.get_tf_root(),
                                      check=False)
        apartment_setup.set_json_goal('KeepHandInWorkspace',
                                      map_frame='map',
                                      base_footprint='base_footprint',
                                      tip_link=tcp)
        goal_point = PointStamped()
        goal_point.header.frame_id = 'iai_apartment/cabinet1_door_top_left'
        # apartment_setup.set_json_goal('DiffDriveTangentialToPoint',
        #                               goal_point=goal_point)
        apartment_setup.set_avoid_joint_limits_goal(50)
        apartment_setup.plan_and_execute()

        apartment_setup.set_json_goal('Open',
                                      tip_link=tcp,
                                      environment_link=handle_name,
                                      goal_joint_state=goal_angle)
        apartment_setup.set_json_goal('DiffDriveTangentialToPoint',
                                      goal_point=goal_point)
        apartment_setup.set_avoid_joint_limits_goal(50)
        apartment_setup.plan_and_execute()

        apartment_setup.set_json_goal('Open',
                                      tip_link=tcp,
                                      environment_link=handle_name,
                                      goal_joint_state=0)
        apartment_setup.set_json_goal('DiffDriveTangentialToPoint',
                                      goal_point=goal_point)
        apartment_setup.set_avoid_joint_limits_goal(50)
        apartment_setup.plan_and_execute()

    def test_dishwasher(self, apartment_setup: TiagoTestWrapper):
        dishwasher_middle = 'dishwasher_drawer_middle'
        base_pose = PoseStamped()
        base_pose.header.frame_id = dishwasher_middle
        base_pose.pose.position.x = -1
        base_pose.pose.position.y = -0.25
        base_pose.pose.orientation.w = 1
        base_pose = apartment_setup.transform_msg(apartment_setup.default_root, base_pose)
        base_pose.pose.position.z = 0
        apartment_setup.set_seed_odometry(base_pose)

        tcp = 'gripper_left_grasping_frame'
        handle_name = 'handle_cab7'
        handle_name_frame = 'handle_cab7'
        goal_angle = np.pi / 2
        left_pose = PoseStamped()
        left_pose.header.frame_id = handle_name_frame
        left_pose.pose.position.x = -0.1
        left_pose.pose.orientation = Quaternion(*quaternion_from_matrix([[1, 0, 0, 0],
                                                                         [0, -1, 0, 0],
                                                                         [0, 0, -1, 0],
                                                                         [0, 0, 0, 1]]))
        apartment_setup.set_cart_goal(left_pose,
                                      tip_link=tcp,
                                      root_link=apartment_setup.default_root,
                                      check=False)
        apartment_setup.plan_and_execute()

        apartment_setup.set_json_goal('Open',
                                      tip_link=tcp,
                                      environment_link=handle_name,
                                      goal_joint_state=goal_angle)
        # apartment_setup.set_json_goal('KeepHandInWorkspace',
        #                               map_frame='map',
        #                               base_footprint='base_footprint',
        #                               tip_link=tcp)
        # apartment_setup.allow_all_collisions()
        apartment_setup.plan_and_execute()
        # apartment_setup.set_apartment_js({joint_name: goal_angle})

        apartment_setup.set_json_goal('Open',
                                      tip_link=tcp,
                                      environment_link=handle_name,
                                      goal_joint_state=0)
        # apartment_setup.set_json_goal('KeepHandInWorkspace',
        #                               map_frame='map',
        #                               base_footprint='base_footprint',
        #                               tip_link=tcp)
        # apartment_setup.allow_all_collisions()
        apartment_setup.plan_and_execute()
        # apartment_setup.set_apartment_js({joint_name: 0})

    def test_hand_in_cabinet(self, apartment_setup: TiagoTestWrapper):
        tcp = 'gripper_left_grasping_frame'
        handle_name_frame = 'iai_apartment/cabinet1'
        left_pose = PoseStamped()
        left_pose.header.frame_id = handle_name_frame
        # left_pose.pose.position.x = 0.1
        left_pose.pose.orientation = Quaternion(*quaternion_from_matrix([[-1, 0, 0, 0],
                                                                         [0, 0, -1, 0],
                                                                         [0, -1, 0, 0],
                                                                         [0, 0, 0, 1]]))
        apartment_setup.set_cart_goal(left_pose,
                                      tip_link=tcp,
                                      root_link=tf.get_tf_root(),
                                      check=False)
        apartment_setup.plan_and_execute()


class TestConstraints:
    def test_DiffDriveTangentialToPoint(self, apartment_setup):
        goal_point = PointStamped()
        goal_point.header.frame_id = 'iai_apartment/cabinet1_door_top_left'
        apartment_setup.set_json_goal('DiffDriveTangentialToPoint',
                                      goal_point=goal_point)
        apartment_setup.plan_and_execute()


class TestJointGoals:
    def test_out_of_joint_soft_limits3(self, zero_pose: TiagoTestWrapper):
        js = {
            'arm_right_5_joint': -2.1031066629465776,
        }
        zero_pose.set_seed_configuration(js)
        zero_pose.set_joint_goal(zero_pose.better_pose2, check=False)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()
        zero_pose.are_joint_limits_violated()

    def test_out_of_joint_soft_limits4(self, zero_pose: TiagoTestWrapper):
        js = {
            'arm_right_5_joint': -3,
        }
        zero_pose.set_seed_configuration(js)
        zero_pose.set_joint_goal(js, check=False)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()
        zero_pose.are_joint_limits_violated()

    def test_out_of_joint_soft_limits5(self, zero_pose: TiagoTestWrapper):
        js = {
            'arm_right_5_joint': 3,
        }
        zero_pose.set_seed_configuration(js)
        zero_pose.set_joint_goal(js, check=False)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()
        lower_limit, upper_limit = zero_pose.world.joints['arm_right_5_joint'].get_limit_expressions(0)
        assert lower_limit < zero_pose.world.state['arm_right_5_joint'].position < upper_limit

    def test_out_of_joint_soft_limits6(self, zero_pose: TiagoTestWrapper):
        js = {
            'arm_left_1_joint': 0.2999719152605501,
            'arm_left_2_joint': -1.0770103752381537,
            'arm_left_3_joint': 1.538071509678013,
            'arm_left_4_joint': 1.9890333283428183,
            'arm_left_5_joint': -1.9048274775275487,
            'arm_left_6_joint': 1.1571632014060012,
            'arm_left_7_joint': 0.6375186679031444,
            'arm_right_1_joint': 0.2716512459628158,
            'arm_right_2_joint': -1.0826349036723986,
            'arm_right_3_joint': 1.5215471970784948,
            'arm_right_4_joint': 2.0190831292184264,
            'arm_right_5_joint': -1.9978066473844511,
            'arm_right_6_joint': 1.206231348680554,
            'arm_right_7_joint': 0.4999018438525896,
            'gripper_left_left_finger_joint': 0.0016879097059911336,
            'gripper_left_right_finger_joint': 0.0017554347466345558,
            'gripper_right_left_finger_joint': 0.0009368327712725209,
            'gripper_right_right_finger_joint': 0.01085319375968001,
            'head_1_joint': -0.0033940217264349197,
            'head_2_joint': -0.9843060924802811,
            'torso_lift_joint': 0.3487927046680741,
        }

        # zero_pose.set_seed_configuration(js)
        # zero_pose.set_joint_goal(zero_pose.better_pose2, check=False)
        pointing_goal = PointStamped()
        pointing_goal.header.frame_id = 'base_footprint'
        pointing_goal.point.x = 2
        zero_pose.set_pointing_goal(tip_link='xtion_link',
                                    root_link='base_footprint',
                                    goal_point=pointing_goal)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()
        zero_pose.are_joint_limits_violated()

    def test_out_of_joint_soft_limits7(self, zero_pose: TiagoTestWrapper):
        js = {
            'arm_left_1_joint': 1.3505632726981545,
            'arm_left_2_joint': -1.1195635667275154,
            'arm_left_3_joint': 2.3115915820828667,
            'arm_left_4_joint': 0.9410957928690423,
            'arm_left_5_joint': -0.9138386896689713,
            'arm_left_6_joint': 0.32143255957945216,
            'arm_left_7_joint': 2.0158598934576375,
            'arm_right_1_joint': 0.3006285274060041,
            'arm_right_2_joint': -1.1107991645139517,
            'arm_right_3_joint': 1.4956685979283315,
            'arm_right_4_joint': 2.0495483917627206,
            'arm_right_5_joint': -1.9978918685519071,
            'arm_right_6_joint': 1.2007401412818746,
            'arm_right_7_joint': 0.49898664261947634,
            'gripper_left_left_finger_joint': 0.04418634626215283,
            'gripper_left_right_finger_joint': 0.04378708138690458,
            'gripper_right_left_finger_joint': 0.03060437169265785,
            'gripper_right_right_finger_joint': 0.04051188814220821,
            'head_1_joint': 0.018108434658135126,
            'head_2_joint': 0.09081672674822555,
            'torso_lift_joint': 0.3486754163734459,
        }

        zero_pose.set_seed_configuration(js)
        js2 = deepcopy(js)
        # del js2['arm_right_2_joint']
        zero_pose.set_joint_goal(js2, check=False)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()
        zero_pose.are_joint_limits_violated()

    def test_joint_goals(self, zero_pose: TiagoTestWrapper):
        js1 = {
            'arm_left_1_joint': - 1.0,
            'arm_left_2_joint': 0.0,
            'arm_left_3_joint': 1.5,
            'arm_left_4_joint': 2.2,
            'arm_left_5_joint': - 1.5,
            'arm_left_6_joint': 0.5,
            'arm_left_7_joint': 0.0,
        }
        js2 = {
            'arm_right_1_joint': - 1.0,
            'arm_right_2_joint': 0.0,
            'arm_right_3_joint': 1.5,
            'arm_right_4_joint': 2.2,
            'arm_right_5_joint': - 1.5,
            'arm_right_6_joint': 0.5,
            'arm_right_7_joint': 0.0,
        }
        zero_pose.set_joint_goal(js1)
        zero_pose.set_joint_goal(js2)
        zero_pose.plan_and_execute()

    def test_joint_goals_at_limits(self, zero_pose: TiagoTestWrapper):
        js1 = {
            'head_1_joint': 99,
            'head_2_joint': 99
        }
        zero_pose.set_joint_goal(js1, check=False)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()
        zero_pose.set_joint_goal(zero_pose.default_pose)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()

    def test_SetSeedConfiguration(self, zero_pose: TiagoTestWrapper):
        zero_pose.set_json_goal('SetSeedConfiguration',
                                seed_configuration=zero_pose.better_pose)
        zero_pose.set_joint_goal(zero_pose.default_pose)
        zero_pose.plan()

    def test_SetSeedConfiguration2(self, zero_pose: TiagoTestWrapper):
        zero_pose.set_json_goal('SetSeedConfiguration',
                                seed_configuration=zero_pose.better_pose)
        zero_pose.set_joint_goal(zero_pose.default_pose)
        zero_pose.plan_and_execute(expected_error_codes=[MoveResult.CONSTRAINT_INITIALIZATION_ERROR])

    def test_get_out_of_joint_soft_limits(self, zero_pose: TiagoTestWrapper):
        js = {
            'head_1_joint': 1.3,
            'head_2_joint': -1
        }
        zero_pose.set_json_goal('SetSeedConfiguration',
                                seed_configuration=js)
        zero_pose.set_joint_goal(zero_pose.default_pose)
        zero_pose.plan()

    def test_torso_goal(self, zero_pose: TiagoTestWrapper):
        # js1 = {
        #     'torso_lift_joint': 0.2,
        # }
        # js2 = {
        #     'torso_lift_joint': 0.25,
        #     'head_1_joint': 1
        # }
        # zero_pose.set_joint_goal(js1)
        # # zero_pose.allow_all_collisions()
        # zero_pose.plan_and_execute()
        #
        # zero_pose.set_joint_goal(js2)
        # # zero_pose.allow_all_collisions()
        # zero_pose.plan_and_execute()

        # zero_pose.set_json_goal('SetSeedConfiguration',
        #                         seed_configuration=js_start)
        zero_pose.allow_self_collision()
        js = deepcopy(zero_pose.default_pose)
        # del js['head_1_joint']
        # del js['head_2_joint']
        zero_pose.set_joint_goal(js)
        zero_pose.plan()
        zero_pose.allow_self_collision()
        js = deepcopy(zero_pose.default_pose)
        del js['gripper_right_left_finger_joint']
        # del js['gripper_right_right_finger_joint']
        zero_pose.set_joint_goal(js)
        zero_pose.plan()

    def test_get_out_of_joint_soft_limits2(self, zero_pose: TiagoTestWrapper):
        js = {
            'head_1_joint': 2,
            'head_2_joint': -2
        }
        js2 = {
            'head_1_joint': 2.1,
            'head_2_joint': 0,
            'torso_lift_joint': 1.5
        }
        zero_pose.set_json_goal('SetSeedConfiguration',
                                seed_configuration=js)
        zero_pose.set_joint_goal(js2, check=False)
        zero_pose.plan()
        zero_pose.are_joint_limits_violated()