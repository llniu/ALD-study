/*
Formål:
At regne andre ting ud til Lili som hvor lang tid der i gennemsnit gik til folk fik deres event og sådannoget. 

Detaljer: 
Der skal anvendes datasættet Lilis_markører_hepProg

Inspiration
20200315a_HarrellsC_med_CI_hepProg fra mappen med dofiler til den prognostiske artikel. 

*/





cd "W:\Nglefiler\Ditlevs Projekt\202003 Forår 2020\2020_8 Lili Nius artikel\Datasæt"


clear
use Lilis_markører_hepProg

codebook id
//Her er de 358, som Lili gav os data på. 

*Hvor mange af lillis patienter soreterer vi væk, når vi fjerner dem uden follow up
codebook id if bopæl == 2
//10 af Lilis patienter kan vi ikke bruge fordi de ikke har boet i vores region. Ingen Follow-up. 


*Der laves den nye st-setting
keep if bopæl == 1 | bopæl == 3 //Dem uden follow-up fjernes
*Studytime er den tid en patient er i studiet. Hvis der ikke sker nogen event er det fra inklusion og ind til ceosurering. Hvis der sker en event, er det fra inklusion og ind til eventen sker. 

codebook id
//348 patienter tilbage

gen studytime = cens_tid - incl if hepProg == 0
replace studytime = date - incl if hepProg == 1
sort id date //Nødvendig for de næste kommandoer
*Hvis der ikke er nogen event, så beholder vi kun en redord. Studytime er der jo kodet for. 
egen noevent = tag(id) if hepProg ==0
drop if noevent ==0 &hepProg == 0
//Der efterlades dog en 0-event også ved de id-numre, der fik en event. 
*Hvis folk har fået flere events, er det kun den første, der skal beholdes. 
egen event = tag(id) if hepProg == 1
drop if event ==0 &hepProg == 1
*Kun én record per patient og det skal være den, hvor der sker en event. 
bysort id: egen max_event = max(hepProg)
keep if max_event == hepProg
drop noevent event max_event
*Nu kan der st-settes. 
stset studytime,  failure(hepProg==1)

*Der er 348 patienter med. 70 har hepProg

*vi ændrer tidsvariablen til måneder
gen _t_temp = _t/30.4
drop _t 
rename _t_temp _t

*blandt de 70, der får hepProg, hvor lang tid går der så før de får det?
tabstat  _t if _d == 1, statistics(median p25 p75) 
//mean time to event 17 months (IQR 5-30)

*Hvor lang er vores follow-up på alle patienter
tabstat  _t , statistics(median p25 p75) 
//mean follow-up 43 months (IQR 21 -60)
