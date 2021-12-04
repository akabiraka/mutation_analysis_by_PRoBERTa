import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

run_1_train_losses=[0.7856312990188599, 0.7788035869598389, 0.7830132246017456, 0.7826519012451172, 0.7845069766044617, 0.7789196968078613, 0.7816028594970703, 0.7808718085289001, 0.7855859994888306, 0.7792440056800842, 0.78645920753479, 0.784899890422821, 0.7828865051269531, 0.7808288335800171, 0.7804332375526428, 0.7816671133041382, 0.78439861536026, 0.7834681272506714, 0.783541202545166, 0.7794868350028992, 0.7785138487815857, 0.7771605253219604, 0.7868897914886475, 0.7818247079849243, 0.7816579341888428, 0.7807241678237915, 0.7794848680496216, 0.785194993019104, 0.7862059473991394, 0.7852427959442139, 0.781937301158905, 0.7833754420280457, 0.7898430228233337, 0.7780065536499023, 0.7820714116096497, 0.7850411534309387, 0.7828018665313721, 0.782659113407135, 0.783673107624054, 0.7844170928001404, 0.7848060727119446, 0.7816805243492126, 0.7823110818862915, 0.7833089828491211, 0.781940221786499, 0.7831582427024841, 0.7824221849441528, 0.7835119962692261, 0.7841718792915344, 0.7799298167228699]
run_1_val_losses=[1.1526556015014648, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738677740097046, 0.4738682806491852, 0.4738709628582001, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374]
run_2_train_losses=[0.7856312990188599, 0.7788035869598389, 0.7830132246017456, 0.7826519012451172, 0.7845069766044617, 0.7789196968078613, 0.7816028594970703, 0.7808718085289001, 0.7855859994888306, 0.7792440056800842, 0.78645920753479, 0.784899890422821, 0.7828865051269531, 0.7808288335800171, 0.7804332375526428, 0.7816671133041382, 0.78439861536026, 0.7834681272506714, 0.783541202545166, 0.7794868350028992, 0.7785138487815857, 0.7771605253219604, 0.7868897914886475, 0.7818247079849243, 0.7816579341888428, 0.7807241678237915, 0.7794848680496216, 0.785194993019104, 0.7862059473991394, 0.7852427959442139, 0.781937301158905, 0.7833754420280457, 0.7898430228233337, 0.7780065536499023, 0.7820714116096497, 0.7850411534309387, 0.7828018665313721, 0.782659113407135, 0.783673107624054, 0.7844170928001404, 0.7848060727119446, 0.7816805243492126, 0.7823110818862915, 0.7833089828491211, 0.781940221786499, 0.7831582427024841, 0.7824221849441528, 0.7835119962692261, 0.7841718792915344, 0.7799298167228699]
run_2_val_losses=[1.1526556015014648, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738677740097046, 0.4738682806491852, 0.4738709628582001, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374]
run_2_val_losses=[1.1526556015014648, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738677740097046, 0.4738682806491852, 0.4738709628582001, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374]
run_3_train_losses=[0.7819836735725403, 0.779193103313446, 0.7827063202857971, 0.7816517949104309, 0.7874453663825989, 0.7784330248832703, 0.7806232571601868, 0.7826828360557556, 0.7851917147636414, 0.7861223816871643, 0.7843596339225769, 0.7845993041992188, 0.7826148867607117, 0.7886399030685425, 0.7807305455207825, 0.7866161465644836, 0.7868009209632874, 0.782518744468689, 0.7833729982376099, 0.7813047766685486, 0.7837634682655334, 0.7839404344558716, 0.7871106266975403, 0.7829660773277283, 0.7795599699020386, 0.7881214022636414, 0.786871612071991, 0.7820559740066528, 0.7906700372695923, 0.7876896858215332, 0.7853816151618958, 0.784692645072937, 0.7846211791038513, 0.7839192748069763, 0.7806269526481628, 0.7785840630531311, 0.7807745337486267, 0.783395528793335, 0.7822571992874146, 0.7830797433853149, 0.7804588675498962, 0.7836200594902039, 0.7839874029159546, 0.7792590260505676, 0.7860672473907471, 0.786780834197998, 0.7835608124732971, 0.7825689315795898, 0.7854523062705994, 0.7862225770950317]
run_3_val_losses=[0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738677442073822, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374]
run_4_train_losses=[0.7674452662467957, 0.7766108512878418, 0.7769169807434082, 0.7683678865432739, 0.7811819314956665, 0.7816709876060486, 0.7838555574417114, 0.7838709354400635, 0.7832238078117371, 0.787524938583374, 0.7851408123970032, 0.7837988138198853, 0.7791870832443237, 0.7801447510719299, 0.7822373509407043, 0.781747579574585, 0.783315896987915, 0.7847775816917419, 0.7812317609786987, 0.7806801199913025, 0.7838529348373413, 0.78609299659729, 0.7863997220993042, 0.7851752042770386, 0.7835045456886292, 0.7831971049308777, 0.7838894724845886, 0.7863401770591736, 0.7816706895828247, 0.782223105430603, 0.7811709046363831, 0.7856918573379517, 0.7874183654785156, 0.7842508554458618, 0.7847087383270264, 0.7835718393325806, 0.7854098081588745, 0.7802808284759521, 0.7775810360908508, 0.7825024724006653, 0.7877798676490784, 0.7873369455337524, 0.783916711807251, 0.7818740606307983, 0.7860984802246094, 0.7876074314117432, 0.7808940410614014, 0.7858507633209229, 0.7802145481109619, 0.7825493812561035]
run_4_val_losses=[0.6495774388313293, 0.47386792302131653, 0.4738725423812866, 0.4738679826259613, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374]
run_5_train_losses=[0.7815503478050232, 0.7860344052314758, 0.7838455438613892, 0.7807647585868835, 0.7848063111305237, 0.7799003720283508, 0.7859143614768982, 0.7838677763938904, 0.78835529088974, 0.7762146592140198, 0.7779493927955627, 0.7827827334403992, 0.790420651435852, 0.7813515663146973, 0.7879311442375183, 0.7826604247093201, 0.7816459536552429, 0.7800097465515137, 0.7816311717033386, 0.7820460796356201, 0.7842594981193542, 0.7858614921569824, 0.7799665927886963, 0.7810052037239075, 0.7783597111701965, 0.7828211784362793, 0.7839229702949524, 0.7825252413749695, 0.7867559194564819, 0.7877436876296997, 0.7839730381965637, 0.7793666124343872, 0.7850456237792969, 0.7779315114021301, 0.7886059880256653, 0.7820025682449341, 0.7846307158470154, 0.7805234789848328, 0.7815453410148621, 0.7831403613090515, 0.777822732925415, 0.7844622135162354, 0.7866131663322449, 0.7876932621002197, 0.7821513414382935, 0.7821131944656372, 0.7804548144340515, 0.7846090793609619, 0.7882433533668518, 0.7822561860084534]
run_5_val_losses=[0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 0.4738676846027374, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648]
run_6_train_losses=[0.7784526348114014, 0.7740710377693176, 0.7770691514015198, 0.7862232327461243, 0.781944751739502, 0.7827973365783691, 0.7815959453582764, 0.7940036654472351, 0.7795771360397339, 0.7845106720924377, 0.7696669101715088, 0.7888262867927551, 0.7820088863372803, 0.7799726724624634, 0.7839190363883972, 0.7871766090393066, 0.7823016047477722, 0.7794451117515564, 0.7767033576965332, 0.7830624580383301, 0.7730271816253662, 0.782008171081543, 0.781853973865509, 0.7711005806922913, 0.7718585133552551, 0.7768828272819519, 0.7825801372528076, 0.7817640900611877, 0.7866986393928528, 0.7816004753112793, 0.7835253477096558, 0.7782204747200012, 0.7782975435256958, 0.7786702513694763, 0.7803242802619934, 0.7849673628807068, 0.7828536629676819, 0.781750500202179, 0.7816444039344788, 0.7820169925689697, 0.7783050537109375, 0.7865275740623474, 0.78472900390625, 0.783099353313446, 0.787671685218811, 0.7862663865089417, 0.7831634879112244, 0.7812278866767883, 0.7868401408195496, 0.7814972996711731]
run_6_val_losses=[1.1526556015014648, 1.1526556015014648, 0.47460779547691345, 0.47386792302131653, 0.5063336491584778, 0.48467105627059937, 0.4828081429004669, 0.4817206561565399, 0.482699990272522, 0.48276472091674805, 0.48069649934768677, 0.4803043603897095, 0.4785277843475342, 0.48054781556129456, 0.4787479043006897, 0.47925958037376404, 0.47980862855911255, 0.4785686135292053, 0.4793015718460083, 0.47719836235046387, 0.47899556159973145, 0.4776287376880646, 0.47707563638687134, 0.4773629307746887, 0.47715067863464355, 0.47702839970588684, 0.4762924909591675, 0.4746011793613434, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648, 1.1526556015014648]


def plot_losses(train_losses, validation_losses, runs, filepath=None):    
    for i, run_no in enumerate(runs):
        plt.plot(train_losses[i], label="{} train loss".format(run_no))
        plt.legend(loc='best')
        plt.plot(validation_losses[i], label="{} validation loss".format(run_no))
        plt.legend(loc='best')
        plt.xlabel("Epochs")
        plt.ylabel("Cross-entropy loss")
    if filepath is not None:    
        plt.savefig(filepath, dpi=300, format="pdf", bbox_inches='tight', pad_inches=0.0)
        plt.cla()
    else: plt.show()
    
plot_losses([run_1_train_losses, run_2_train_losses, run_3_train_losses, run_4_train_losses, run_5_train_losses, run_6_train_losses],
            [run_1_val_losses, run_2_val_losses, run_3_val_losses, run_4_val_losses, run_5_val_losses, run_6_val_losses], 
            ["run-1", "run-2", "run-3", "run-4", "run-5", "run-6"],
            filepath="outputs/images/model_analysis/train_val_loss.pdf")    