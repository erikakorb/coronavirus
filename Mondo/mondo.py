import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wget
import os
from scipy import optimize
import datetime                             #data
from matplotlib.dates import DateFormatter



######## FONTI ######

# John Hopkins University, con dati da tutto il mondo resi disponibili su github su     https://github.com/CSSEGISandData/COVID-19
# un tizio li ha poi riformattati e reimpaginati per poterli scaricare bene su          https://github.com/datasets/covid-19
# ma dal 4/4/2020 ha avuto problemi nell'aggiornarli pertanto prendo i dati dalla john hopkins direttamente (3 pagine raw su github)
#
# aggregatore dati disponibili sui tamponi                                              https://ourworldindata.org/covid-testing
#
# popolazione nazioni principali                                                        https://it.wikipedia.org/wiki/Stati_per_popolazione



#raccolgo ed estraggo dati

f=open('dati_totali.csv','w+')              
os.remove('dati_totali.csv') 
url_tot='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv' 
wget.download(url_tot, './dati_totali.csv') 
dati_totali=pd.read_csv('dati_totali.csv',sep=',')              #totali

f=open('dati_morti.csv','w+')              
os.remove('dati_morti.csv') 
url_morti='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv' 
wget.download(url_morti, './dati_morti.csv') 
dati_morti=pd.read_csv('dati_morti.csv',sep=',')              #morti

f=open('dati_guariti.csv','w+')              
os.remove('dati_guariti.csv') 
url_guariti='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv' 
wget.download(url_guariti, './dati_guariti.csv') 
dati_guariti=pd.read_csv('dati_guariti.csv',sep=',')              #guariti



datipop=np.genfromtxt('popolazione.txt', usecols=1)          #popolazione per nazione 


#curve da fittare
def logistica(x,a,b,c,d):
    return a/(np.exp(-b*x)+d)+c

def log_asintoto(a,c,d):
    return a/d+c

def logistica_derivata(x,a,b,d):
    return a*b*np.exp(-b*x)/((np.exp(-b*x)+d)**2)


#seleziono i dati relativi ad una nazione

exp=2  #giorni in più esponenziale
log=15 #giorni in più logistica

def selezione(ID_nazione): #NB:ID_nazione è una stringa
    if ID_nazione == 'France':
        nazione_tot=dati_totali.loc[dati_totali['Lat'] == 46.2276]
        nazione_morti=dati_morti.loc[dati_morti['Lat'] == 46.2276]
        nazione_guariti=dati_guariti.loc[dati_guariti['Lat'] == 46.2276]
        #tamponi=datitamp.loc[datitamp['Entity'] == 'France - units unclear']

    elif ID_nazione == 'United Kingdom':
        nazione_tot=dati_totali.loc[dati_totali['Lat'] == 55.3781]
        nazione_morti=dati_morti.loc[dati_morti['Lat'] == 55.3781]
        nazione_guariti=dati_guariti.loc[dati_guariti['Lat'] == 55.3781]
        #tamponi=datitamp.loc[datitamp['Entity'] == 'United Kingdom - people tested']

    elif ID_nazione == 'Netherlands':
        nazione_tot=dati_totali.loc[dati_totali['Lat'] == 52.1326]
        nazione_morti=dati_morti.loc[dati_morti['Lat'] == 52.1326]
        nazione_guariti=dati_guariti.loc[dati_guariti['Lat'] == 52.1326]
        #tamponi=0
    else:
        nazione_tot=dati_totali.loc[dati_totali['Country/Region'] == ID_nazione] #seleziono i dati relativi ad una sola nazione
        nazione_morti=dati_morti.loc[dati_morti['Country/Region'] == ID_nazione]
        nazione_guariti=dati_guariti.loc[dati_guariti['Country/Region'] == ID_nazione]
        

    totali=nazione_tot.values[0,4:]      #a partire dall'indice 4 di colonna ho i dati giorno per giorno
    guariti=nazione_guariti.values[0,4:]
    morti=nazione_morti.values[0,4:]

    attuali=totali-(guariti+morti)
    
    N=np.size(totali)
    giorni=np.arange(N)
    domani=np.arange(N+exp)
    domani_log=np.arange(N+log)

    #stimatore affidabilità dati         
    percentuale=100*morti[48:N]/totali[48:N]     #comincio dal giorno in cui in italia lockdown N=48 10/3/2020

    #giorno zero inizio pandemia NELLA nazione (criterio:ntot>=20 casi perchè devo togliere diamond princess ecc.)
    for i in range(1,N):
        while totali[i] < 20:  #se ho almeno 20 pazienti inizio pandemia
            caso_uno=totali[i]
            i=i+1
        else:
            caso_uno=totali[i] #quanti pazienti positivi il primo giorno
            giorno_uno=i #indice come di un array, quindi i giorni dal 22/1/2020
            break

    #derivata dy/dx con dx=1 (differenza giorni)
    dtot=np.arange(N)           #dy totali
    dattuali=np.arange(N)       #derivata casi attualmente in cura
    dmorti=np.arange(N)
    for i in range (1,N):
        dtot[i]=totali[i]-totali[i-1]
        dattuali[i]=attuali[i]-attuali[i-1]
        dmorti[i]=morti[i]-morti[i-1]

    dtot_corretti=dtot           
    for i in range (1,N):
        if dtot[i-1]<=0:              #correggo grossolanamente bug john hopkins
            dtot_corretti[i]=dtot[i]/2.
            dtot_corretti[i-1]=dtot_corretti[i]
            if dtot[i-2]<=0:
                dtot_corretti[i-2]=dtot_corretti[i]
    
    #derivata seconda dy'/dx con dx=1 (differenza giorni)
    ddtot=np.arange(N)         #ddy totali
    for i in range (1,N):
        ddtot[i]=dtot[i]-dtot[i-1]


    righe=[totali,guariti,morti,percentuale,giorni,giorno_uno,dtot,ddtot,attuali,dattuali,dmorti, dtot_corretti]  #lista

    return righe



#raccolgo i dati in una matrice
nazioni_lista=['Italy','Germany','United Kingdom', 'Spain','France','US','Austria']  #india,france, belgium
matrice_lista=[]

Nnazioni=len(nazioni_lista)
Ngrafici=7   #normalmente 6

for i in range (0,Nnazioni):
    riga=selezione(str(nazioni_lista[i]))       #ogni riga della matrice è una nazione
    matrice_lista.append(riga)

matrice=np.array(matrice_lista)           #6 righe x 13 colonne

#estraggo colonne contenenti i dati di una stessa quantità per nazioni diverse
totali_lista=matrice[:,0]           
guariti_lista=matrice[:,1]
morti_lista=matrice[:,2]
percentuale_lista=matrice[:,3]
giorni=matrice[0,4]         #la colonna 4 ha valori tutti uguali
dtot_lista=matrice[:,6]
ddtot_lista=matrice[:,7]
attuali_lista=matrice[:,8]
dattuali_lista=matrice[:,9]
dmorti_lista=matrice[:,10]
dtotcorretti_lista=matrice[:,11]

N=np.size(giorni)
futuro=N+exp
fut_log=N+log
domani=np.arange(N+exp)
domani_log=np.arange(N+log)



#media mobile
passo=7
w = [1.0/passo]*passo


#date
base=datetime.date(2020,1,22)
date_list=[base+datetime.timedelta(days=x) for x in range(N)]
formatter=DateFormatter('%d %b')
xlim=[date_list[39],date_list[N-1]]    # primo marzo
xlim2=[date_list[223],date_list[N-1]]  # primo settembre
#primo_mese=[date_list[6],date_list[37],date_list[67]]


#preparo plot   
colori_scatter=['green','deepskyblue','mediumpurple','crimson','black', 'dimgrey','fuchsia']   #,'gold',,'black','springreen', violet belgium
colori_plot=['limegreen','cyan','purple', 'red','grey','sienna','pink']    #'gold','grey',
marker_lista=['*','+','.','x','x','o','+']              #'d','d','*'
label_lista=['Italia','Germania','UK','Spagna','Francia','USA','Austria']      #'India','Francia
#label_spaziofasi=['1/04','8/04','15/04']





############# sandbox


######################## PLOT ###########################################

#derivata prima totali grezzi media mobile
fig_dermobile=plt.figure(figsize=(15,10))
ax1=fig_dermobile.add_subplot(1,1,1)
ax1.set_title('Nuovi casi giornalieri (dati grezzi ricalcolati con media mobile)')
ax1.set_ylabel(r'Nuovi casi totali giornalieri (media mobile)')
ax1.xaxis.set_major_formatter(formatter)

for i in range(0,Ngrafici):
    med_dtot=np.convolve(dtot_lista[i],w,"valid")          #creo i dati mediati
    ax1.plot_date(date_list[-len(med_dtot):], med_dtot, color=colori_scatter[i], marker='None', linestyle='solid' , alpha=0.8, label=label_lista[i])
ax1.axhline(y=0,color='grey',linestyle='dashed')

ax1.set_xlim(xlim2)
ax1.set_ylim(bottom=0, top=350000)
ax1.legend(loc=2)
ax1.grid(ls=':')
plt.show()
fig_dermobile.savefig('dtot_mobile.pdf')


#derivata prima totali grezzi NORMALIZZATI per popolazione + media mobile
fig_derpopmobile=plt.figure(figsize=(15,10))
ax1=fig_derpopmobile.add_subplot(1,1,1)
ax1.set_title('Nuovi casi giornalieri (dati grezzi ricalcolati con media mobile ed espressi per 100 000 abitanti)')
ax1.set_ylabel(r'Nuovi casi totali giornalieri per 100 000 abitanti')
ax1.xaxis.set_major_formatter(formatter)

for i in range(0,Ngrafici):
    dtotN=100000*dtot_lista[i]/datipop[i]           #normalizzo i dati grezzi
    med_dtotN=np.convolve(dtotN,w,"valid")          #creo i dati mediati
    ax1.plot_date(date_list[-len(med_dtotN):], med_dtotN, color=colori_scatter[i], marker='None', linestyle='solid', alpha=0.8, label=label_lista[i])
ax1.axhline(y=0,color='grey',linestyle='dashed')

ax1.set_xlim(xlim2)
ax1.set_ylim(bottom=0)
ax1.legend(loc=2)
ax1.grid(ls=':')
#plt.show()
fig_derpopmobile.savefig('dtot_pop_mobile.pdf')


#spazio fasi logaritmico
fig_spaziofasi=plt.figure(figsize=(15,10))
ax1=fig_spaziofasi.add_subplot(1,1,1)
ax1.set_title('Spazio delle fasi logaritmico (dati grezzi)\n Partendo dalla settimana 4 marzo - 10 marzo 2020')
ax1.set_ylabel('Derivata (nuovi casi totali giornalieri)')
ax1.set_xlabel('Casi totali')

for i in range(0,Nnazioni):
    jtot=totali_lista[i]
    jdtot=dtot_lista[i]
##    jtot_mascherato=np.ma.masked_where(jtot==0,jtot)      #jdtot_mascherato[48:N]
##    jdtot_mascherato=np.ma.masked_where(jdtot==0,jdtot)
    jtot_medio=(np.add.reduceat(jtot, np.arange(0, jtot.size, 7)))/7.     #sommo a gruppi di n elementi
    jdtot_medio=(np.add.reduceat(jdtot, np.arange(0, jdtot.size, 7)))/7.  #e poi divido, cioe' medio
    ax1.plot(jtot_medio[6:-1], jdtot_medio[6:-1], color=colori_scatter[i], marker='.',alpha=0.7, label=label_lista[i])
    # indice 6 è media dati settimana mercoledì 4/3 - martedì 10/3


ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.legend(loc=2)
ax1.grid(ls=':')
plt.show()
fig_spaziofasi.savefig('spaziofasi.pdf')


#derivata prima morti
fig_dmorti=plt.figure(figsize=(15,10))
ax1=fig_dmorti.add_subplot(1,1,1)
ax1.set_title('Derivata prima morti')
ax1.set_ylabel(r'Nuovi morti giornalieri')
ax1.xaxis.set_major_formatter(formatter)

for i in range(0,Ngrafici):
    med_dmorti=np.convolve(dmorti_lista[i],w,"valid")
    ax1.plot_date(date_list[-len(med_dmorti):], med_dmorti, color=colori_scatter[i], marker='None', linestyle='solid', alpha=0.8 ,label=label_lista[i])

ax1.axhline(y=0,color='grey',linestyle='dashed')
ax1.set_xlim(xlim2)
ax1.set_ylim(bottom=0, top=450)
ax1.legend(loc=2)
ax1.grid(ls=':')
plt.show()
fig_dmorti.savefig('dmorti.pdf')


#derivata prima morti/popolazione
fig_dmorti=plt.figure(figsize=(15,10))
ax1=fig_dmorti.add_subplot(1,1,1)
ax1.set_title('Morti giornalieri per 100 000 abitanti')
ax1.set_ylabel(r'Nuovi morti giornalieri')
ax1.xaxis.set_major_formatter(formatter)

for i in range(0,Ngrafici):
    dmorti_pop=100000*dmorti_lista[i]/datipop[i]
    med_dmortipop=np.convolve(dmorti_pop,w,"valid")
    ax1.plot_date(date_list[-len(med_dmortipop):], med_dmortipop, color=colori_scatter[i], marker='None', linestyle='solid', alpha=0.8 ,label=label_lista[i])

ax1.axhline(y=0,color='grey',linestyle='dashed')
ax1.set_xlim(xlim)
ax1.set_ylim(bottom=0)
ax1.legend(loc=2)
ax1.grid(ls=':')
plt.show()
fig_dmorti.savefig('dmorti_pop.pdf')


