# About the residue alignment
---

### Alignment two sequence residue

##### **there are following steps to achieve alignment:**

###### 1、set appropriate parameter 

**I select blosum62 scoring matrix and obtain several results according to different gap penalty. It is worth mentioning that some of these results are same just score is different while others are completely different.**

###### 2、construct the scoring matrix H and path matrix D

**For two residue sequence, H is a 2-dimension matrix and formed by these two sequence residue. Each value in the H is a best optimized score(using blosum62 and surrounding area of D to determine the different gap penalty ) from 3 directions(4, 2, 3 stands for left, up, top left) and it take the minimum value of 0.**

**Also, D is a 2-dimension matrix  and formed by these two sequence residue. Each value in the D is a path direction (4, 2, 3 stands for left, up, top left)  by which can gain the best optimized score.**

###### 3、traceback 

**At first, traverse the entire H matrix to find maximum.  After that trace back along the path (according to the D matrix) to find the whole path until arrive 0 value in the H matrix. Ultimately Gain the similar residue sequence from the path**.

---

### Alignment three sequence residue

##### Most steps are like the two sequence residue alignment. The difference is H or D is not a matrix and it can be regarded as a cube (the extension of 2-dimension) formed by three residue sequence.

**1、Three sequence evaluation is sum of two residue sequence one by one  ( score R-S-V = score(R-S) + score(S-V) + score(R-V) )and what have to be aware of is two gaps score is 0 ( score(- -) = 0 ).**

**2、H is a 3-dimension cube and  each value is a best optimized score(using blosum62 and surrounding area of D to determine the different gap penalty ).**

**3、D is a 3-dimension cube and each value in the D is a path direction (1, 2, 3, 4, 5, 6, 7 stands for vector(0, 0, 1)、(0,1,0))  by which can gain the best optimized score.**

---

### Results

##### I get several group results by testing different gap penalty and find most of time  is used to construct H and D. Particularly the results (CPU : I7-6700K RAM : 16GB IDE : Pycharm) are as follows:

**1、two sequence alignment**

```
phkrs_lysRS and Yeast_lysRS:
Similar Sequence:
A: HWADY-IADKIIRERGEKEKYV--VE-SGITPSGYVH---VGNFRELFTAYIVGHALRDKGYEVRHIHMWDDY----D------RFR-K--VPRN--VPQEW--K----DYL------GMP-I---------SE-V-PD-----PW-G-C----------HE--SY--AE-----H--------FMRK-FEEEVEKLGIEVDL-L-YASELYKRGEYSEE--IRLA--FEKRDKIMEILNKY-REIA--KQP---PLPENWWPAMVYCPEHRREAEI---IEWDGGWKVKYKCP-EGHEGWVDIRSGNVKLRW---RV---D------WP-MRWSHFGVDFEPAGKDHLVAGSSY-DT-GKEI-IK-EVY-GKEAPLSLMYEFVGIKGQNVILL-S-DLYEVLEPGLVRFIYARHRPNKEIK----IDLG-----LG--IL------NLY---------D--E----------------------FE--K------------VERIYFG-VE------G---------EELR-RTYELSMPKKPERLVAQAPFRFLAVLVQLPHLTEE--DI--IN-VLIK--QG-HI-------PRDLSKE-D--------VERVKLRINLARN--------WVKKYAPEDVKFSILEKP------------PEVEVSEDV-REAMNEVAEWLENHEEFSVEEFNNILF----EVAKRRGISS-----REWFS--T--L-Y-R--LFIGKER-----GPRLASFLASLDRSFVIKRL
Start 1     Stop 503
B: HP-DVIIVD-LMRN------YIQE-ESSKI--SG-VDSLII------FPA------L-----E------WTNTERGDDLIPIPRRLRKKANNPKDAVV-Q-WEKKPCGDDFLKVEANGG-PIIFFFNPQFLAAKVVPPDLTRKEEDGGCCLVENKKVIIE-ESSSPIAAKFHAGHHRSTIIGGFFLANYYE----KLGWEV-IMMYY---L---GDWGKQGLL-LAGFFE-R---------YNNEEAVKKDPHHL-L---FD---VYV----R---IKDIIEEEG-DSI----PEE--QSTN----G--KAREFKRRMDGDDEALKIWWKFFRE--FSI--E---K--------YDDTAAR-LIIKDDVYGG-ESQVS--------K-ES--MLAADDLFK--EKGL----T--H----EDKAVLIIDLTFNKKLLGAIIVKSDGTTTLYTRDVGAAMDDYEEYHFDKMIYVIASQQDLHAAQFFFELKKMGFEWAKDLQHVVN---FGVVQMSTRKGGVVFLDNILEEETKKKMHEV-M-KKNE--N-----KY-A---QIEH-PEEADDLGIISVVMIQMQQGRRINYEFKWEERMLSFEDDGPYLQYAHHSR--LR-SVERNSGITQEKWWIN--A--D--FSLLKEPAKLLIRLLGQYPP------DVRR---N--A--IKTHEPTTVVTY---LFLTHQQV------SSYDVLW--WVAQTTELLTTRRALLY-GAARVLYNGGMRL---LG-LT---PVERM
Start 25     Stop 606
Total Score: 796.0     Taking:      first_cost = 2      repeat_cost = 0
Time cost of construct HD:     1.7099997997283936 s
Time cost of traceback:      0.032000064849853516 s
```

```
phkrs_lysRS and Thermophilus_gluRS
Similar Sequence:
A: VVESGITPS--G-YVHVGNFRELFTAYI----------VGHALRDKGYEVRHIHMWDDYDRFRKVP----R-----NVPQEWKDYLGMPIS--EVPD---PWGCHESYAEHFMRKFEEEVEKLGIEVDLL--YASELYKRG-EYSEEIRLAFEKRDKIME-I---LNK--Y----REIAKQPPLPENWWPAMVYCPEHR-REAE--IIEWDGGWKVK-YKCPEGHEGWV-D-IRSGNVKLRWRV-DW---P----MR----WS-HF--GVDFEPAGKDHLVAGSSYDTGKEII--KE--VYGKEAPLS-LMYE-FVGIKGQNVILLSDLYEVLEPGLVRFIYARHRP---N--K-EI---K----IDLGLGILNLYDEFEKVERIYFG--VEGEELRR-----TYELSMP--KK--P-ERLVAQAPF---RFLAVLVQL--P--HLTEE---DIINVLIKQG-HIPRD-LSKEDV-ERVK--LR-INLARNWVKK-Y---A-----PE-D-VK-FSILEKP----PEVE--VSE----DVREA---MNE--V---A--EW----LE---NHEEFSVEEFNNI-LFEVAK--R---RGISSRE-WFSTLYR-L-FIGKERGPRLASFLASLDRSFVIKRL-R-L
Start 21     Stop 505
B: VVTR-IAPSTGGPP-HVG------TAYILFNYAWARRNNGG--R---FIVR-I---EDTDRARYVPAEERRLAALKK----W---LGL--SDEEGPDGGPP---HGPY-----R--QSE--RL---P-LYKYYAEELLKRGAAY----R-AFETPEEL-EIIKEK--KGYYGRARRNI---PP--E--E-A-----EERRRRGEHVVI------RLKPPR-P-GT-TEVDDLLR-G-V-V---VDDNEIPPVVLLLKDGYPPTHHLNVVVD------DHLM-GV-TD----VIAEEELVVST---PIHLLLYRFF----G---------WEA--P---RF-Y--HMPLRNNDKKKKIKRKKHTSLLDW-------Y----KAE----GLPPEA--LRNLCLMGGF--SMPGRREFTTEEEFI-QA-FWERR-----VSLGPPFDDL-EKRWM--MN-----GYYI-RELLSLEEVEERVKFLLRAAGLS--WESEYYRRAAELMRPPRDDLLKFFP--EKAYLFTTE-DPVVSEAQRKKLEEGPLLLKEYPPLRAAEEEWEAALLELLRRG--FAAEK--GVLLGQVAQLRRALTTG--SLEPP--GLFELLLLLGKER--------A-L-R-----RLRRLL
Start 2     Stop 467
Total Score: 763.0     Taking:      first_cost = 2      repeat_cost = 0
Time cost of construct HD:     1.2689998149871826 s
Time cost of traceback:      0.025000333786010742 s
```

```
Yeast_lysRS and Thermophilus_gluRS
Similar Sequence:
A: MISQLKKLSIAEPAVAKDSH--PD-----V------N----IV-----DLMR-NYI--SQELSKISGVDSSLIFPALEWTNTM---ERG-DL--LIPIPRLRIKGANP-KDLAVQWAEKFPCGDFL-EK---VE--ANGPFIQF-F--F-NPQFLAKLVIPD-ILTRKE----DYGSCKLVENKKVIIEFSSPNIAKP------FHA--G--HL-RSTIIGGFLANLYEKL---G-WEVIRMNYLGDWGKQF-GLLAVGFERYGNEE-----ALVK-D--P-IH-------HLF---DVYVRINKDIEEE--GDSIP----LEQSTNGKAREYFKR-MEDGDEEALKIWK--RFREFSIEKY-IDTY-ARLNIKYDVYSGESQVSKESMLKA---IDLFK-EKG-LTHEDKGAVLIDLTKFNKKLGKAIVQKSDGTTLYLTRDVGAAM-D-RYEKYHFDKMIYVIASQQDLHAAQFFEILKQMGFEWAKDLQHVNFGMVQGMSTRKGTVVFLDNILEETKEKMHEV-MKKNENKYAQI-E----HPEEVAD-----L--VGI---S------AV--M------IQDM-QGK-RINNY---E-FK-WE---RMLSFEG-DTGPYL-Q-YAHSRLRSVERNASGITQEKW----INADFSLLK--EPAA-K-LLIRLLGQ--YPDVLRNAIKTH--E-PTTVVTYLFKLTHQVSSCYDVLWVA--GQTEELATARLALYGAARQVLYNGMRLLGLTPVER
Start 6     Stop 605
B: MV--VTR--IA-PSPTGDPHGTTAIALFNNAWARRNNGRFIIVIEDTDD--RRRYVGAAEE--RI------LA--ALKWL-GLYDEE-GDDVGP----PH----G--PRR----QS-ERLP----LQQKAEE-ELKKRG----WYYAFFTTPE---EL---EII--RKEGGYDD-GRAR---N----I----P----PEAEERA--ARGGPHHVRR----------L--KVRPGGTTEV-K-----DE---LGGVV-V----YDNQEPDVVL-LLKDDYPPYYHANVVDDHHLMVTDDV-IRA-----EELVV-STPHVLLL----------Y--RFF--G-------WEPRRF-------YMM--PLLR-N--PD----KTKISKR---KSTSLLDWYKEE-GLLP-E---A----L-R-N----------------YLCL-MGFSMDDRR-E-------IF------TLE--EF--I--Q-AFTW----ERVSLG---------G-PVF-D--LE----KLR--MM--N-GKY--IEELSLE--EEVAEVKPFLLEAAGLWESSAYLRRAAVLMMPRFDTLLKEFEE-KRR---YFTEEYYPSSEAQRRKLE-EGP---PLLEEYYP--RLR-----A----QEEWEAALLEA---LLRFA--AAKKVV--KL-GQAQQP--LRAAL-TGLEEPPG-----LF----------EIL--ALGGK--E----R-AL----R-------RL------ER
Start 1     Stop 465
Total Score: 764.0     Taking:      first_cost = 2      repeat_cost = 0
Time cost of construct HD:     1.5760002136230469 s
Time cost of traceback:      0.028999805450439453 s
```

**2、three sequence alignment**

```
Similar Sequence:
A: ----WAD-Y-IADKIIRERGEKEKYVVESGIT-PSGY-VH-VGNFRELFTA---YIVGH---------ALRDKGYEVRHIHM-W----DDYD--RF----RKV--P---RNVP--QE-----W-KDYLGMPIS--EVP-D---P--WGCHESY--AEHF-MRKFEEEV-EKLGIEVDLLYASELYKRGE--YSEEIRLAF---EKRDKIM--E-ILNKYR-EIAK------QPPLPE--NW-------WP--AMVYC------P---EH--RREAE---I--I-EWD-GGWKVKYKCPEGHEGWVDIRS-G----------N------VKL-RWRV-D-WPMRWSHFG--VDFEPAGKDHLVAGSS-YDTGK-EIIK--E--VYGKEAP----LS-------LMYE-FV----GIKGQNVILLSDL-YEVLEPGLVR-FI---YARHRPNKEIKIDL-GLGIL-N--LYDE---FE-KV--E---R----I-YF-----GVE--GEE----------L------RR---T-Y---E--L--SMP--KKPER---------------L---VAQAPF-----R-------FLAVLVQLPHLTEEDIINV--LIKQG---H------IPRD--LSKEDV--ERVK--LRIN--LARNW-VKKY---A------PE----D---VKFSILE-KPPEVE---VS--ED--VRE-AMNEVAEW---LENHEE-FSVEE-FNNIL---FEVAKR-RG------ISSRE-W----FSTLYRLFIG----KE---R--G----PR-L--A----S-----------F---------------------LA----SL---DRSFVIK----RL-------R-L
Start 2     Stop 505
B: MISQLKKLS-IA--EP---------AVAKD-SHP---DVNIV---DL----MRNYISQELSKISGVDSSL-I--F---P-ALEWT--NTM-E--R-GDLL--I--P-----IP----R-L--RIK---G---A---NPKDLA--VQWA--------EKFPCGDF----LEK---V-E---AN-----GP--FIQF----FFNPQFLAKLVIPD-I--LTRKED----Y-GSCKL-VE--NKKVIIEFSSP----NIAK-----PFHAGHL-RSTIIGGFLANLYEKL-G-WEV-IRM-NYLGDWG--KQFGLLAVGFERYGNEE-----ALV-K---D--PI---H-------------HL---FDVYVRINKD-IE-EE----GDSIP----LEQSTNGKAREY--FKRMEDGD----EEAL--KIWKRF-----REFSIEKY------ID---T-YA-R-L-NIK-YDVYSG-ESQVSKESMLKA---IDLFKEKGLTHEDKGAVLIDLTKFNKKLGKAIVQKSDGTTLYLTRD--VGAAM-D-RY-EKYHFDKMIYVIASQQDLH--AAQ--F-FEILK-QMG---FE-WAKDLQHV-NFG------MV-QGMSTRKGTVVFL--DNIL--EE-TKE--K--M-HEV-MKKN-ENK-Y---AQIE--HPEEVA-DLVGISAVMIQ-DM-QGK-R-IN---NY---E----F--KWERMLSF--EGD--TGPY---L-Q-YAH-SRLRSVERNASGITQEKWINADFS-LLKE-P--AA-KL-LIRLLGQ--YPDVLRNAIKTH--E-PTTVVTYLFKLTHQVSSCYDVLWVAGQTEELATARLALYGAAR-QVL-YNGMRLLGLTPVER-M
Start 6     Stop 606
C: MV-----VTRIA--------------P----S-PTG-DPH-VG------TA---YI------------AL----F-N----YAWARRN----GGRF--IVR-IEDTDRARYVPGAEERILAA-LK-WLGL--SYDEGP-DVGGPHGP-----YRQSERLPL--Y-----QK--------YAEELLKRG-WAY----R-AFETPE---EL---EQI----RKE--KGGYDG------RARN----I---PPEEAEE--RARRGEP----HVIR-------L-KVPR--PGTTEV--K--D----E--LR--GVV-V----YDNQEIPDVV-LLK---SDGYP--TYHLANVVD------DHLM-G--VT-----DVIRAEEWLVST---PIHVLL----------YRAF-----G-------------WE--AP---R-F----Y--H---------MP---LLRN---PD-----KTKISKR---KSHTSLDWYK-----AE--G--------F---LPEAL--R----N-YL---CLMGFSMPDGR--E-------IF------TLEEF-IQA-FTWE--RVSLGGPVF-----DLEKL--------RWM--NG----K----YI-RE-VLSLEEV-AERVKPFLR-EAGL--SWESEAYLRRA-VELMRPR---FD--TLK----EF--PE-KARYL-FTEDYPVSEKA------Q-RKL---EEG--L--P---LLKELY---PRLR-----A----QEEWTEAALEALLR---GFAAEK-GV-KL-GQVAQP--LRAAL-TGSLETPG-----LF-------------EI------LA---L-L-GKERA--LR----RL------ERAL
Start 1     Stop 467
Total Score: 2437.0     Taking:      first_cost = 2      repeat_cost = 0
Time cost of construct HD:     3029.8689999580383 s
Time cost of traceback:      10.740000009536743 s
```

```
Similar Sequence:
A: VVES-G---IT-PS---G-YVH--VGN-F--RELFTAYI-VGHALR-DKG-YEVRHIHMWDDYDRFRKVP----RNV-PQ-EWKDYLGMPISEV-PD-PWGCHESYAEHFM-RK-FEEEV----EKL-GIEVDLLY---ASELYKRGE--YSEEIRLAF---EKRDKIM--EIL-NK----Y----RE-IAKQPPLP-ENW-WP--AMVYC-PEHR---REAEI-IEWDGGWKVKYKCPEGHEGWVD-IRSGNVKLRWRV-DW----P----MRWS-HF---GVDFEPA--GKD---HLVAGSS-YDTGK-EIIK--E--VYGKEAPL--S-----LMYE-F----VGI-KGQNV-----ILLSDLYEVLEPG--L-VRFI-YARHRP--NKEI--K----IDL--GL-GILNLYDE-FEK--V-ERI--YF---G---V---EGEELRRTYELSMPKKP--ERLVAQAPFRF-LAVL-V-----QLPHLTE--EDIINVLI---K--Q----GH--IPRD-LSKEDV-ERVK-----LRIN--LARNWVK-KY---APE-----DVKFS-ILE-KPPE-----VEVS----ED-VREAMN--EV-AEW--LENHEE-FSVEE-FNNI--LFEVAKR-RG------ISSRE-W----FSTLYRL-FIG-KE---R--G----P---R--L-----AS------FL--ASLDRSF----VI-K--R-L---RLE----G
Start 21     Stop 507
B: MISQLKKLSIAEPAVAK-DS-HPDV-N--I-VDLMRNYISQELS-K-ISG---V-------DSS----L-------IFPALEW-TN-TM---ERG-DLLI--PI-PRLRIKGANP-KDLAVQWAEKFPCG-D-FL-EKV--EA--NGP--FIQF--F-FN-PQFLAKLVIPDILTRKED--Y-GSCKLVENKKVII--EFS-SPNIA-K---PFHAGHLRS-TI-I---GGFLAN-L-YE-KLGW-EVIR--MNYLG----DWGKQF---GLLAV--GFERYGNE-E-A-LVKDPIHHL---FDVYVRINKD-IE-EE----GDSIPLEQSTNGKAREY--FKRMEDGDEEALKIWKRFREFSIEKYI--DTYARLNIKYDVYS-GESQVSKESMLKA---IDLFKE-KG-L-TH-E--DKGAVLIDLTKFNKKLGKAIVQKSDG---TTLY-LTRDVGAAMDRY--EK-YHFDKMIY-VIASQQDL-HAAQFFEILKQMGFEWAKDLQHVNFGM--V--QGMS--T---R-KGTVVFL-DNI-LEET--KEK-MH---EVMKKNENKYAQI-E-H-PEEVADLVGISAVMIQDMQGKRINNYEFK-WERMLSF--EGD--TGPY--L-Q-YAHS-RLRSVERNASGITQEKWINADFS-LLKEP-AA-KL-LIRLLGQ--YPDVLRNAIKTHEPTTVVTY-LFKLTHQVSSCYDVLWVAGQTEE-LATARL--ALYG
Start 6     Stop 585
C: MV--VTR--IA-PSP-TGDP-H--VGTAYIA--LF-NYAW---ARRN-GGRFIVR-I--E-DTDRARYVPGAEER-ILAALKW---LGLSYDE-GPDV--G---GP--H--G--PYR----QS-ERLP------LYQKYAEELLKRG-WAY----R-AFETPE---EL---EQI-RKEKGGYDG--R-A--RN--IPPE-EAEER-AR-RGEP-H-V-IR---LKV-PRPGT-----T-EV--KD-E-LR-G-V-V---VYDN-QEIPDVVLLK-SDGY--PT--YHLANVV-D--DHLM-G--VT-----DVIRAEEWLVST---PI--H--VLL--YRAF-----G-WEA---P-RF-------Y-HM-P---L-LR--NPDKTK--ISKR---KSHTSLDWYK-AEGFL--P-EAL-R--NY--L--C--LMGFSMP---DGRE---IFTL--------EEFI-QA-FTWE----RV--S---LGG-P-VFD-L-----E--K-LRWMN-G-KYI-REVLSLEEVAERVK---PFLR-EAGL--SWESEAYLRRAVELMR---PRFDTLKEF--PEK-ARYL-FT----EDY--P-VS--E-KA-QRKL---EEG--L--PL--LKELY--P-RLR-----A----QEEWTEAALEALLR-GFAAEK-GV-KL-GQVAQP--LRAAL-TG---SLETPGLF--E--I---L-AL-L-GK-ERALR--RLERAL-A
Start 1     Stop 468
Total Score: 1946.0     Taking:      first_cost = 2      repeat_cost = 1
Time cost of construct HD:     2997.1500000953674 s
Time cost of traceback:      13.352999925613403 s
```

```
Similar Sequence:
A: VVES----GIT-PS----GYVHVGN-F-RELFTAYI-VGHALR-DKG-YEVRHIHMWDDYDRFRKVP----RNV-PQ-EWKDYLGMPISEV-PD-PWGCHESYAEHFM-RK-FEEEV----EKL-GIEVDLLY---ASELYKRGE--YSEEIRLAF---EKRDKIM--EILNKYR-EIAK---Q-PPLP--ENW-------WP--AMVYC------PEH--RREAE---I--I-EWD-GGWKVKYKCPEGHEGWVDIRSG----------N------VKL-RWRVD-WPMRWSH-F--GVDFEPAGKDHLVA-GSSYDTGKEI-IKE--VYGKE-AP-LSLMYE-F----VGI-KGQNV-----ILLSDLYEVLEPG--L-VRFI-YARHRP--NKEI--K----IDL--GL-GILNLYDEFEK-V-ERIYF-----G---V---EGEELRRTYELSMPKKP--ERLVAQAPFRF----LAVLVQ--LPHLTE--EDIINVLI---K--Q----GH--IPRD-LSKEDV-ERVK-----LRIN--LARNWVK-KY---APE-----DVKFS-ILE-KPPE----VEVS----ED-VREAMN--EVAEW--LENHEE-FSVEE-FNNILFEVA-KR-RG------ISSRE-W----FSTLYRL-FIG-KE--R--G----P---R--L-----AS------FL--ASLDRSF-VI----K-R-L---RLEG
Start 21     Stop 507
B: MISQLKKLSIAEPAVAKDSHPDV-N--IVDLMRNYISQELS-K-ISG---V-------DSS----L-------IFPALEWTNT--M---ERG-DLLI--PI-PRLRIKGANP-KDLAVQWAEKFPCG-D-FL-EKV--EA--NGP--FIQF----FFNPQFLAKLVIPDIL--TRKED----Y-GSC-KLVENKKVIIEFSSPNIAKPFH-A--G---HL-RSTIIGGFLANLYEKL-G-WEV-IRM-NYLGDWGK-QFGLLAVGFERYGNEE-----ALVK---D--PI--HHLFDVYV--RIN-KD-IEEEG---D---SIPLEQ---STNGKARE----Y--FKRMEDGDEEALKIWKRFREFSIEKYI--DTYARLNIKYDVYS-GESQVSKESMLKA---IDLFKE-KG-L-THED-KGAVLIDLTKFNKKLGKAIVQKSDG---TTLY-LTRDVGAAMDRY--EK-YHFDKMIYVIASQQDL-HAAQFFEILKQMGFEWAKDLQHVNFGM--V--QGMS-----TR-KGTVVFL-DNI-LEET--KEK-MH---EVMKKNENKYAQI-E-HPEEVADLVGISAVMIQDMQGKRINNYEF-KWERMLSFE-GD--TGPY---L-QYAHSRLRSVERNASGITQEKWINADFS-LLKEP-AA-KLLIRLLGQ--YPDVLRNAIKTHEPTTVVTY-LFKLTHQVSSCYDVLWVAGQTEELATARLAL
Start 6     Stop 583
C: MV--VTR--IA-PSP--TGDPHVGTAYI-ALF-NYAW---ARRN-GGRFIVR-I---EDTDRARYVPGAEER-ILAALKW---LGLSYDE-GPDV--G---GP--H--G--PYR----QS-ERLP------LYQKYAEELLKRG-WAY----R-AFETPE---EL---EQI---RKE--KGGYDG----RARN----I---PPEEAEE--RARRGEP-HVIR-------L-KVPR--PGTTEV--K--D----E--LR-GVV-V----YDNQEIPDVV-LLKS--DGYP--TYHLANV-VD------DHL--MGV-TD-V--IRAEEWLVST--PI-HV-LLYRAF-----G-WEA---P-RF-------Y-HM-P---L-LR--NPDKTK--ISKR---KSHTSLDWYK-AEGFL--PEALR--NY--LC-L---MGFSMP---DGRE---IFTL--------EEFI-QA-FTWE--------RVSLGG-P-VFD-L-----E--K-LRWMN-G-KYI-REVLSLEEVAERVK---PFLR-EAGL--SWESEAYLRRAVELMR---PRFDTLKEF-P-EKARYL-FT----EDY--P-VS--EKA-QRKL---EEG--L--P---LLKE-LYPRLR-----A----QEEWTEAALEALLR-GFAAEK-GVKL-GQVAQP--LRAAL-TG---SLETPGLF--E--I---LALL---GKERALR--RLE-
Start 1     Stop 464
Total Score: 1624.0     Taking:      first_cost = 3      repeat_cost = 1
Time cost of construct HD:     2670.707000017166 s
Time cost of traceback:      8.672999858856201 s
```

```
Similar Sequence:
A: VVESGIT-PSGY--VHVG-NFRELFTAYI-VGHALRDKGYEVRHIH--M-WDDY-DRF-RKVP--R-NVPQEW-KDYLGMPISE-VP--D--PWG-CHESYAEHFMRKFEEEVEKLGIEVDLLYASELYKRGE-YSEEIRLAFEKRDKIMEILNKYREIAKQPP----L--PENWWP-AMV-YCPEHRREA-EII--EWDGGWKVKY-----KCPE-GH-EGWV-D-IRS------GNVK-LRWRVDW-PMRWSHFGV--D-FEPA--GKDHLVAGSSYDTGKEIIK-EVY-GKE-APLSLMYE-FVGIKGQNVILLSDLYEVLEPG-LVRFIYARHRPNKE-IKIDLG-----LGILNLY--D-EF--EKVER--IYF-G---VEGEELRRTYE--LSMPKKPERLVAQAPFRFLAVLVQLPHLTEEDI--INVLIKQGH-IPRD---LSKEDV-ERVK--LR-INLARNW-VKKY---A-----PE---D-VKFSILEKPPEVE---VS--ED--VRE-AM-N-EVAE--W-LEN-HEEF-SVEEFNNILFEVAKRRGISSREW----FSTL--YR--LFI---GKERGPRLASFLASLDRSFVIKRLRLEGK
Start 21     Stop 508
B: LKKLSIAEPAVAKDSHPDVNIVDLMRNYISQELS-KISGVDSSLIFPALEWTNTMERGDLLIPIPRLRIKGANPKD-LAVQWAEKFPCGDFLEKVEANGPFI-QFFFNPQFL-AKLVIP-------DILTRKEDYGSC-KLV-ENKKVIIEFS--SPNIAK-PFHAGHLRSTIIGGFLA-NLY--EKL--GWEVIRMNYLGDWGKQFGLLAVGFERYGNEEALVKDPIHHLFDVYVRINKDIEEEGDSIPLEQSTNGKAREYFKRMEDGDEEAL--KIWKRFRE-FSIEKYI-DTYARLNIKYDVYSG-ESQ-VSKE----SMLKAIDLFKEKGLTHE-DKGAVLIDLTKFNKKLGKAIVQKSDGTTLY--LTRD----VGAAM-DRYEKYH-FDKMIYVIASQQDLHA-AQF-F-EILKQMGFEWAKDLQHVNFGMVQGMS-TRKGTVVFLDNILEETKEKMHEV-MKKN-ENK-Y---A-QIEHPEEVADLVGISAVMIQ-DMQGKRINNYE-FK-WER-MLSFEGDTG-PYLQYAHSRLRSVER--N-----AS--GITQEKWINADFSLLKEPAAKLLIRLLG--QYP---DVL----RN-AIK--THEPT
Start 10     Stop 547
C: VVT-RIA-PSPTGDPHVGTAYIALF-NYAW---ARR-NG-G-RFI-VRIEDT---DR-ARYVPGAEERILAA-LK-WLGLSYDEG-P--D-V--GGPHGPY-RQS----E----RLPL-YQK-YAEELLKRGWAY----RA-FETPEEL-E------QIRKEK---GGYDG------RARNIP-PE---EAEERARR---GEP---H-VIRLKVPRPGTTE--VKDELRGVV-VYD--NQEI-P--DVVLLK-SD-G--YPTYHLANVVDDHLM-G-VT----DVIRAEEWLVSTPIHVLL-YRAF-GWEAPRF------Y-HM-P--LLR-----N-PDK--TKISKRKSHTSLD---WYKAEG-FLPEAL-RNYLCLMGFSMPDGRE---IFT--L------EEF-IQA-FTW-ERV-SLGG-PVFDLEKLRW-M-NG-KYIRE--VLSLEEVAERVKPFLREAGL--SWESEAYLRRAVELMRPR--FDTLK-E-F---PE-KARYL-FTEDYPVSEKAQRKLE--EGLPLLKELYPRLRAQEEWTEAALE-ALLRGFAAEKGVK--LGQVAQP---LRAALTGSLETPGLFEILALLGKERALRRL--E-R
Start 2     Stop 465
Total Score: 943.0     Taking:      first_cost = 5      repeat_cost = 2
Time cost of construct HD:     2886.941999912262 s
Time cost of traceback:      13.281000137329102 s
```



### Analysis and Summary

##### According to Smith-Waterman algorithm, every value in H can get by the following formula:

<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;$$&space;F(i,j)&space;=&space;max&space;\begin{cases}&space;0\\&space;F(i-1,j-1)&plus;s(x_i,y_j)\\&space;F(i-1,j)-d\\&space;F(i,j-1)-d&space;\end{cases}&space;$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\large&space;$$&space;F(i,j)&space;=&space;max&space;\begin{cases}&space;0\\&space;F(i-1,j-1)&plus;s(x_i,y_j)\\&space;F(i-1,j)-d\\&space;F(i,j-1)-d&space;\end{cases}&space;$$" title="\large $$ F(i,j) = max \begin{cases} 0\\ F(i-1,j-1)+s(x_i,y_j)\\ F(i-1,j)-d\\ F(i,j-1)-d \end{cases} $$" /></a>

##### I extend this to 3-dimension, but the time complexity O(n) is multiply. I test the time cost of construct HD when they are matrices is about 1.5 S and speculate the cost when cubes is more than 1.6 * 7/3 * 500 = 1866.7 S. Actually the time to construct HD cubes is more than 2500 S after testing.

##### To accelerate the alignment, Altschul put forward a BLAST that can produce less but better point to increase the accuracy. Proceed as follows:

**1、Find out from the two sequences some subsequences of equal length that can form a perfect match without gaps, that is, sequence fragment pairs.**

**2、Find all pairs of sequence fragments between two sequences that match more than a certain value.**

**3、The obtained sequence fragment pair is extended according to a given similarity threshold value to obtain a similarity fragment of a certain length, which is called a high score fragment pair**

##### Specific how to realize can use Blast or its API and I don't write blast source code repeatedly.

@author	FanYu

@Date	2019/12/03
