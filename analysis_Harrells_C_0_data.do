/*
Opgave danne et datasæt, der er kan bruges til at beregne Harrell's C til Lili

Hvad skal datasættet indeholde?
-multiple records per person
-endepunkt er hepProg
-prædiktorerne er alle NIlt'er, fibrosegrad te, swe osv Her bruger vi vores data
-Lilis proteomics prædiktorer.

Indledende arbejde med data modtaget fra Lili 
-samle alle hendes "logistic-værdier" i én datafil
-Koble denne til Id-nummer. 
*/


clear
*De data, som Lili sendte før sommerferien med et file-share program ligger på denne sti: 
cd "W:\Nglefiler\Ditlevs Projekt\202003 Forår 2020\2020_8 Lili Nius artikel\Datafiler modtaget fra Lili"

*Lili har sendt sine data som csv-filer. De skal laves om til -dta-filer for at vi kan bruge dem. Og data skal samles i én kolonne. //kDe lægges i en særlig mappe. 




*vi indlæser Lilis data om F2_marker_prot_logistic

import delimited F2_prot_Logistic
describe
//Vi skal have samlet værdierne for training og test. 
generate f2_prot_værdi = . , after(sampleid)
replace f2_prot_værdi = f2_prot_logistic_y_test_pred if f2_prot_logistic_y_test_pred !=.
replace f2_prot_værdi = f2_prot_logistic_y_train_pred if f2_prot_logistic_y_train_pred !=. & f2_prot_værdi ==.

keep sampleid f2_prot_værdi

cd "W:\Nglefiler\Ditlevs Projekt\202003 Forår 2020\2020_8 Lili Nius artikel\Lilis filer på dta format"
save f2_prot_værdi, replace


*Vi indlæser og behandler Lilis F3_prot_logistic
clear
cd "W:\Nglefiler\Ditlevs Projekt\202003 Forår 2020\2020_8 Lili Nius artikel\Datafiler modtaget fra Lili"
import delimited F3_prot_Logistic
generate f3_prot_værdi = . , after(sampleid)
replace f3_prot_værdi = f3_prot_logistic_y_test_pred if f3_prot_logistic_y_test_pred !=.
replace f3_prot_værdi = f3_prot_logistic_y_train_pred if f3_prot_logistic_y_train_pred !=. & f3_prot_værdi ==.
keep sampleid f3_prot_værdi
cd "W:\Nglefiler\Ditlevs Projekt\202003 Forår 2020\2020_8 Lili Nius artikel\Lilis filer på dta format"
save f3_prot_værdi, replace


*vi indlæser og behandler Lilis I2_prot_Logistic
clear
cd "W:\Nglefiler\Ditlevs Projekt\202003 Forår 2020\2020_8 Lili Nius artikel\Datafiler modtaget fra Lili"
import delimited I2_prot_Logistic
generate i2_prot_værdi = . , after(sampleid)
replace i2_prot_værdi = i2_prot_logistic_y_test_pred if i2_prot_logistic_y_test_pred !=.
replace i2_prot_værdi = i2_prot_logistic_y_train_pred if i2_prot_logistic_y_train_pred !=. & i2_prot_værdi ==.
keep sampleid i2_prot_værdi
cd "W:\Nglefiler\Ditlevs Projekt\202003 Forår 2020\2020_8 Lili Nius artikel\Lilis filer på dta format"
save i2_prot_værdi, replace


*vi indlæser og behandler Lilis S1_prot_Logistic
clear
cd "W:\Nglefiler\Ditlevs Projekt\202003 Forår 2020\2020_8 Lili Nius artikel\Datafiler modtaget fra Lili"
import delimited S1_prot_Logistic
generate s1_prot_værdi = . , after(sampleid)
replace s1_prot_værdi = s1_prot_logistic_y_test_pred if s1_prot_logistic_y_test_pred !=.
replace s1_prot_værdi = s1_prot_logistic_y_train_pred if s1_prot_logistic_y_train_pred !=. & s1_prot_værdi ==.
keep sampleid s1_prot_værdi
cd "W:\Nglefiler\Ditlevs Projekt\202003 Forår 2020\2020_8 Lili Nius artikel\Lilis filer på dta format"
save s1_prot_værdi, replace


*vi indlæser og behandler Lilis ID-nøgle
clear
cd "W:\Nglefiler\Ditlevs Projekt\202003 Forår 2020\2020_8 Lili Nius artikel\Datafiler modtaget fra Lili"
import delimited "ID key"
keep sampleid participantid patientid cohort group2
cd "W:\Nglefiler\Ditlevs Projekt\202003 Forår 2020\2020_8 Lili Nius artikel\Lilis filer på dta format"
save id_nøgle, replace


*De fire nye .dta-filer skal samles i én -dta-fil, og id-nøglen skal tages i brug, så vikan finde ud af, hvilke patienter i ALD-kohorten, det svarer til. 
clear
cd "W:\Nglefiler\Ditlevs Projekt\202003 Forår 2020\2020_8 Lili Nius artikel\Lilis filer på dta format"
use f2_prot_værdi
merge 1:1 sampleid using f3_prot_værdi
//Komplet match
drop _merge
merge 1:1 sampleid using i2_prot_værdi
//Der er 8 missing
drop _merge
merge 1:1 sampleid using s1_prot_værdi
//Der er 8  missing, også her
drop _merge

*Vi skal også have koblet til ID-nøglen
merge 1:1 sampleid using id_nøgle
//Der er 250 patienter, som godt nok har et ID-nummer, men som ikke har nogen værdi på nogle af Lilis proteomics scorer. Dem sletter jeg
drop if _merge == 2
drop _merge

*De to variable participantid og group 2 behøves ikke for at kunn eidentificere patietnerne. Og Nu er det ikke længere nødvendigt at vide på hvilken plade Lili har analyseret hvad. 
drop participantid group2
drop sampleid

*alle Patienterne er fra den udvidede GALAALD kohorte. Vi skal have patientid-nummeret til at være numerisk, og det skal hede det samme som det gør i vores andre datasæt. 
drop cohort
destring patientid, replace
rename patientid id
order id
sort id


*Nu er de data, Lili er kommet med klar. Næste trind er at de skal merges med datasættet mod_alco9_v5 

cd "W:\Nglefiler\Ditlevs Projekt\201910 Efterår 2019\Databehanding prognostisk artikel\Datasæt"
merge 1:m id using mod_alco9_v5, generate(_merge2)
*Der er nogle observationer som er fra patienter, som Lili ikke har data på disse slettes
drop if _merge2 == 2

*Nu skal vi fokucerer på at reducere antallet af variable. vi skal have lillis Prædiktorer, og vi skal have hepProg. Men ikke ret meget andet. 
keep id f2_prot_værdi f3_prot_værdi i2_prot_værdi s1_prot_værdi redcap_event_name date0 date  incl kleiner elf ft apri forns aar te swe hepProg bopæl cens_årsag cens_tid fib4 p3np cap
describe

*Nu skulle datasættet være klar. Det gemmes i sin egen mappe
cd "W:\Nglefiler\Ditlevs Projekt\202003 Forår 2020\2020_8 Lili Nius artikel\Datasæt"
save Lilis_markører_hepProg