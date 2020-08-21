/*
Form�l:
At beregne den prognostiske styrke af Lilis mark�rer og andre mark�rer

Detaljer: 
Der skal anvendes datas�ttet Lilis_mark�rer_hepProg

Inspiration
20200315a_HarrellsC_med_CI_hepProg fra mappen med dofiler til den prognostiske artikel. 

*/





cd "W:\Nglefiler\Ditlevs Projekt\202003 For�r 2020\2020_8 Lili Nius artikel\Datas�t"


clear
use Lilis_mark�rer_hepProg



*Der laves den nye st-setting
keep if bop�l == 1 | bop�l == 3 //Dem uden follow-up fjernes
*Studytime er den tid en patient er i studiet. Hvis der ikke sker nogen event er det fra inklusion og ind til ceosurering. Hvis der sker en event, er det fra inklusion og ind til eventen sker. 
gen studytime = cens_tid - incl if hepProg == 0
replace studytime = date - incl if hepProg == 1
sort id date //N�dvendig for de n�ste kommandoer
*Hvis der ikke er nogen event, s� beholder vi kun en redord. Studytime er der jo kodet for. 
egen noevent = tag(id) if hepProg ==0
drop if noevent ==0 &hepProg == 0
//Der efterlades dog en 0-event ogs� ved de id-numre, der fik en event. 
*Hvis folk har f�et flere events, er det kun den f�rste, der skal beholdes. 
egen event = tag(id) if hepProg == 1
drop if event ==0 &hepProg == 1
*Kun �n record per patient og det skal v�re den, hvor der sker en event. 
bysort id: egen max_event = max(hepProg)
keep if max_event == hepProg
drop noevent event max_event
*Nu kan der st-settes. 
stset studytime,  failure(hepProg==1)

*Der er 348 patienter med. 70 har hepProg


*F2 proteomics mark�r
stcox f2_prot_v�rdi 
estat concordance
//Harrell's c bliver s� 0.8829

*F3 proteomics mark�r
stcox f3_prot_v�rdi 
estat concordance
//Harrells c bliver s� 0.8563


*TE
stcox te 
estat concordance
//Harrell'S c for TE er 0.8853


*ELF
stcox elf
estat concordance
//Harrell'S c for TE er 0.8570


*2D-SWE
stcox swe
estat concordance
//Harrell's C for 2d-SWE er 0.8743


*FibroTest
stcox ft
estat concordance
//Harrell's C for Fibrotest er 0.8097


*p3np
stcox p3np
estat concordance
//Harrell,s c er 0.8093 for p3np


*Apri
stcox apri
estat concordance
//Harrell's c for apri er 0.7996


*fib4
stcox fib4
estat concordance
//Harrell's c er 0.8170 for Fib-4


*forns
stcox forns
estat concordance
//Harrell's C er 0.7947 for Forns


*Kleiner
stcox kleiner
estat concordance
//Harrell's C er 0.8293 for Kleiner


*AAR
stcox aar
estat concordance
//Harrell's C er 0.7850

*for at v�re grundige pr�ver vi med Lilis steatosemark�r og inflammationsmark�r

*Lilis inflammationsmark�r
stcox i2_prot_v�rdi
estat concordance
//0.8915 ?!

*Lillis steatosemark�r
stcox s1_prot_v�rdi
estat concordance
//0.6586
