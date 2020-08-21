/*
Opgave danne et datas�t, der er kan bruges til at beregne Harrell's C til Lili

Hvad skal datas�ttet indeholde?
-multiple records per person
-endepunkt er hepProg
-pr�diktorerne er alle NIlt'er, fibrosegrad te, swe osv Her bruger vi vores data
-Lilis proteomics pr�diktorer.

Indledende arbejde med data modtaget fra Lili 
-samle alle hendes "logistic-v�rdier" i �n datafil
-Koble denne til Id-nummer. 
*/


clear
*De data, som Lili sendte f�r sommerferien med et file-share program ligger p� denne sti: 
cd "W:\Nglefiler\Ditlevs Projekt\202003 For�r 2020\2020_8 Lili Nius artikel\Datafiler modtaget fra Lili"

*Lili har sendt sine data som csv-filer. De skal laves om til -dta-filer for at vi kan bruge dem. Og data skal samles i �n kolonne. //kDe l�gges i en s�rlig mappe. 




*vi indl�ser Lilis data om F2_marker_prot_logistic

import delimited F2_prot_Logistic
describe
//Vi skal have samlet v�rdierne for training og test. 
generate f2_prot_v�rdi = . , after(sampleid)
replace f2_prot_v�rdi = f2_prot_logistic_y_test_pred if f2_prot_logistic_y_test_pred !=.
replace f2_prot_v�rdi = f2_prot_logistic_y_train_pred if f2_prot_logistic_y_train_pred !=. & f2_prot_v�rdi ==.

keep sampleid f2_prot_v�rdi

cd "W:\Nglefiler\Ditlevs Projekt\202003 For�r 2020\2020_8 Lili Nius artikel\Lilis filer p� dta format"
save f2_prot_v�rdi, replace


*Vi indl�ser og behandler Lilis F3_prot_logistic
clear
cd "W:\Nglefiler\Ditlevs Projekt\202003 For�r 2020\2020_8 Lili Nius artikel\Datafiler modtaget fra Lili"
import delimited F3_prot_Logistic
generate f3_prot_v�rdi = . , after(sampleid)
replace f3_prot_v�rdi = f3_prot_logistic_y_test_pred if f3_prot_logistic_y_test_pred !=.
replace f3_prot_v�rdi = f3_prot_logistic_y_train_pred if f3_prot_logistic_y_train_pred !=. & f3_prot_v�rdi ==.
keep sampleid f3_prot_v�rdi
cd "W:\Nglefiler\Ditlevs Projekt\202003 For�r 2020\2020_8 Lili Nius artikel\Lilis filer p� dta format"
save f3_prot_v�rdi, replace


*vi indl�ser og behandler Lilis I2_prot_Logistic
clear
cd "W:\Nglefiler\Ditlevs Projekt\202003 For�r 2020\2020_8 Lili Nius artikel\Datafiler modtaget fra Lili"
import delimited I2_prot_Logistic
generate i2_prot_v�rdi = . , after(sampleid)
replace i2_prot_v�rdi = i2_prot_logistic_y_test_pred if i2_prot_logistic_y_test_pred !=.
replace i2_prot_v�rdi = i2_prot_logistic_y_train_pred if i2_prot_logistic_y_train_pred !=. & i2_prot_v�rdi ==.
keep sampleid i2_prot_v�rdi
cd "W:\Nglefiler\Ditlevs Projekt\202003 For�r 2020\2020_8 Lili Nius artikel\Lilis filer p� dta format"
save i2_prot_v�rdi, replace


*vi indl�ser og behandler Lilis S1_prot_Logistic
clear
cd "W:\Nglefiler\Ditlevs Projekt\202003 For�r 2020\2020_8 Lili Nius artikel\Datafiler modtaget fra Lili"
import delimited S1_prot_Logistic
generate s1_prot_v�rdi = . , after(sampleid)
replace s1_prot_v�rdi = s1_prot_logistic_y_test_pred if s1_prot_logistic_y_test_pred !=.
replace s1_prot_v�rdi = s1_prot_logistic_y_train_pred if s1_prot_logistic_y_train_pred !=. & s1_prot_v�rdi ==.
keep sampleid s1_prot_v�rdi
cd "W:\Nglefiler\Ditlevs Projekt\202003 For�r 2020\2020_8 Lili Nius artikel\Lilis filer p� dta format"
save s1_prot_v�rdi, replace


*vi indl�ser og behandler Lilis ID-n�gle
clear
cd "W:\Nglefiler\Ditlevs Projekt\202003 For�r 2020\2020_8 Lili Nius artikel\Datafiler modtaget fra Lili"
import delimited "ID key"
keep sampleid participantid patientid cohort group2
cd "W:\Nglefiler\Ditlevs Projekt\202003 For�r 2020\2020_8 Lili Nius artikel\Lilis filer p� dta format"
save id_n�gle, replace


*De fire nye .dta-filer skal samles i �n -dta-fil, og id-n�glen skal tages i brug, s� vikan finde ud af, hvilke patienter i ALD-kohorten, det svarer til. 
clear
cd "W:\Nglefiler\Ditlevs Projekt\202003 For�r 2020\2020_8 Lili Nius artikel\Lilis filer p� dta format"
use f2_prot_v�rdi
merge 1:1 sampleid using f3_prot_v�rdi
//Komplet match
drop _merge
merge 1:1 sampleid using i2_prot_v�rdi
//Der er 8 missing
drop _merge
merge 1:1 sampleid using s1_prot_v�rdi
//Der er 8  missing, ogs� her
drop _merge

*Vi skal ogs� have koblet til ID-n�glen
merge 1:1 sampleid using id_n�gle
//Der er 250 patienter, som godt nok har et ID-nummer, men som ikke har nogen v�rdi p� nogle af Lilis proteomics scorer. Dem sletter jeg
drop if _merge == 2
drop _merge

*De to variable participantid og group 2 beh�ves ikke for at kunn eidentificere patietnerne. Og Nu er det ikke l�ngere n�dvendigt at vide p� hvilken plade Lili har analyseret hvad. 
drop participantid group2
drop sampleid

*alle Patienterne er fra den udvidede GALAALD kohorte. Vi skal have patientid-nummeret til at v�re numerisk, og det skal hede det samme som det g�r i vores andre datas�t. 
drop cohort
destring patientid, replace
rename patientid id
order id
sort id


*Nu er de data, Lili er kommet med klar. N�ste trind er at de skal merges med datas�ttet mod_alco9_v5 

cd "W:\Nglefiler\Ditlevs Projekt\201910 Efter�r 2019\Databehanding prognostisk artikel\Datas�t"
merge 1:m id using mod_alco9_v5, generate(_merge2)
*Der er nogle observationer som er fra patienter, som Lili ikke har data p� disse slettes
drop if _merge2 == 2

*Nu skal vi fokucerer p� at reducere antallet af variable. vi skal have lillis Pr�diktorer, og vi skal have hepProg. Men ikke ret meget andet. 
keep id f2_prot_v�rdi f3_prot_v�rdi i2_prot_v�rdi s1_prot_v�rdi redcap_event_name date0 date  incl kleiner elf ft apri forns aar te swe hepProg bop�l cens_�rsag cens_tid fib4 p3np cap
describe

*Nu skulle datas�ttet v�re klar. Det gemmes i sin egen mappe
cd "W:\Nglefiler\Ditlevs Projekt\202003 For�r 2020\2020_8 Lili Nius artikel\Datas�t"
save Lilis_mark�rer_hepProg