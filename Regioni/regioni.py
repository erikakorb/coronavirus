import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wget
import os
from scipy import optimize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes #zoom
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import datetime                             #data
from matplotlib.dates import DateFormatter, MonthLocator
import mplcursors

#raccolgo ed estraggo dati
f=open('dati.csv','w+')#creo un file vuoto o apro quello già presente. Così che in ogni caso posso far andare il comando successivo
os.remove('dati.csv') #cancello il vecchio file scaricato, se presente, in modo da non avere doppioni con nomi diversi
url='https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv' #ulr con dati aggiornati dalla protezione civile automaticamente ogni giorno, in formato cvs
wget.download(url, './dati.csv') #scarico il file .cvs e lo salvo come dati.cvs

dati=pd.read_csv('dati.csv',sep=',')
dati.fillna(0, inplace=True)    #tolgo i nan e li rendo 0 su tutto il dataframe

datipop=np.genfromtxt('popolazione.txt',int)          #popolazione per regione 	http://www.salute.gov.it/imgs/C_17_monitoraggi_51_2_fileNazionale.pdf
datiPLricov, datiPLti, datiPLtiDL=np.genfromtxt('posti_letto.txt',usecols=(1,2,3), unpack=True)  #ricoverati e T.I.   https://www.agenas.gov.it/covid19/web/index.php?r=site%2Ftab2 
#https://twitter.com/vi__enne/status/1428732053512671233 per le stime dei PL in TI secondo DL 105/2021
# pagina aggiornata https://docs.google.com/spreadsheets/d/e/2PACX-1vSwS7SWNpcv8qy1wZHK7DvWWX7RgDEYzYjzAm_TdExqax3waIuN2Bd0rj2OAh9YRJSDX-QbnqEkfdqJ/pubhtml#
report=str('10/09/2021')   # data stime TI sulla base del report ISS disponibile (data ultima consultazione)
agenas=str('27/01/2022')   # ultimo aggiornamento manuale agenas posti area non critica

def selezione(codice_regione): #NB:ID_regione è una stringa

    if codice_regione == 4: #trentino + bolzano
        trentino =dati.loc[dati['codice_regione'] == 21]
        alto =dati.loc[dati['codice_regione'] == 22]
        regione = pd.concat([trentino, alto]).groupby('data').sum().reset_index()   #concateno i df e li sommo col criterio di righe uguali in colonna 'data'
        print('ciao')

    else:
        regione=dati.loc[dati['codice_regione'] == codice_regione] #seleziono i dati relativi ad una sola regione

    ricoverati=regione.values[:,6]
    intensiva=regione.values[:,7]
    ospedalizzati=regione.values[:,8]
    attuali=regione.values[:,10]
    guariti=regione.values[:,13]
    morti=regione.values[:,14]
    tamp_malati=regione.values[:,15]
    tamp_screening=regione.values[:,16]
    totali=regione.values[:,17]
    tamponi=regione.values[:,18]
    persone=regione.values[:,19]

    N=np.size(totali)

    #derivata dy/dx con dx=1 (differenza giorni)
    dtot=np.arange(N)         #dy totali
    dtot_attuali=np.arange(N)         #dy attualmente positivi
    dtot_morti=np.arange(N)         #dy morti
    dtot_tamponi=np.arange(N)         #dy attualmente positivi
    dtot_persone=np.arange(N)
    dtot_tamp_malati=np.arange(N)
    dtot_tamp_screening=np.arange(N)
    dtot_osp=np.arange(N)
    dtot_intensiva=np.arange(N)
    dtot_ricov=np.arange(N)
    
    for i in range (1,N):
        dtot[i]=totali[i]-totali[i-1]
        dtot_attuali[i]=attuali[i]-attuali[i-1]
        dtot_morti[i]=morti[i]-morti[i-1]
        dtot_tamponi[i]=tamponi[i]-tamponi[i-1]
        dtot_persone[i]=persone[i]-persone[i-1]
        dtot_tamp_malati[i]=tamp_malati[i]-tamp_malati[i-1]
        dtot_tamp_screening[i]=tamp_screening[i]-tamp_screening[i-1]
        dtot_osp[i]=ospedalizzati[i]-ospedalizzati[i-1]
        dtot_intensiva[i]=intensiva[i]-intensiva[i-1]
        dtot_ricov[i]=ricoverati[i]-ricoverati[i-1]

    righe=[totali,guariti,morti,attuali,tamponi,persone,dtot,dtot_attuali,dtot_morti,dtot_tamponi,persone, dtot_persone,
           dtot_tamp_malati,dtot_tamp_screening,tamp_malati,tamp_screening,ospedalizzati,dtot_osp,intensiva,dtot_intensiva,ricoverati,dtot_ricov]
    return righe


dati20=dati.loc[0:20,:] 
##diz={k: g['denominazione_regione'].tolist() for k,g in dati20.groupby('codice_regione')}
diz={k: g['denominazione_regione'].values[0] for k,g in dati20.groupby('codice_regione')}

#raccolgo i dati in una matrice
matrice_lista=[]
Nregioni=len(diz)

for key in diz:                     #qui uso gli indici regionali
    print(key)
    riga=selezione(int(key))        #ogni riga della matrice è una regione
    matrice_lista.append(riga)
    #print(matrice_lista)

matrice=np.array(matrice_lista)           #6 righe x tot colonne

#estraggo colonne contenenti i dati di una stessa quantità per regioni diverse
totali_lista=matrice[:,0]
guariti_lista=matrice[:,1]
morti_lista=matrice[:,2]
attuali_lista=matrice[:,3]
tamponi_lista=matrice[:,4]
persone_lista=matrice[:,5]
dtot_lista=matrice[:,6]
dtot_attuali_lista=matrice[:,7]
dtot_morti_lista=matrice[:,8]
dtot_tamponi_lista=matrice[:,9]
persone_lista=matrice[:,10]
dtot_persone_lista=matrice[:,11]
#regioni_lista=matrice[:,12]
dtot_tamp_malati_lista=matrice[:,12]
dtot_tamp_screening_lista=matrice[:,13]
tamp_malati_lista=matrice[:,14]
tamp_screening_lista=matrice[:,15]
ospedalizzati_lista=matrice[:,16]
dtot_ospedalizzati_lista=matrice[:,17]
intensiva_lista=matrice[:,18]
dtot_intensiva_lista=matrice[:,19]
ricoverati_lista=matrice[:,20]
dtot_ricoverati_lista=matrice[:,21]

N=np.size(totali_lista[0])
giorni=np.arange(N)

#date
base=datetime.date(2020,2,24)
date_list=[base+datetime.timedelta(days=x) for x in range(N)]
formatter=DateFormatter('%d %b')
primo_mese=[date_list[6],date_list[37],date_list[67]]

##settimane=int((N-1)/7)
multipli_7=np.arange(0,N,7)   # individuo i multipli di 7
mask_mar=multipli_7 + 1
if N==mask_mar[-1]:
    mask_mar=mask_mar[:-1]      # sistemo indici

mask_gio=multipli_7 + 3     # primo giovedi 27/2 ha indice 3
if mask_gio[-2]>=N-7:         # e.g. l'indice MASCHERA gio=528 identifica un giovedi 528 in italiatot, quindi quando raggiungo 536-7=529 ancora non mi va bene
    mask_gio=mask_gio[:-2]      # rimuovo l-ultimo elemento se non è ancora passata una settimana
elif N-7 > mask_gio[-2]>=N-11:         # e.g. l'indice MASCHERA gio=528 identifica un giovedi 528 in italiatot, quindi quando raggiungo 536-7=529 ancora non mi va bene
    mask_gio=mask_gio[:-1] 

mask_gio_succ=mask_gio+7    # giovedi successivo


#italia
italia_tot=[ sum(row[i] for row in totali_lista) for i in range(len(totali_lista[0])) ]
italia_dtot=[ sum(row[i] for row in dtot_lista) for i in range(len(dtot_lista[0])) ]
italia_attuali=[ sum(row[i] for row in attuali_lista) for i in range(len(attuali_lista[0])) ]
italia_dtot_attuali=[ sum(row[i] for row in dtot_attuali_lista) for i in range(len(dtot_attuali_lista[0])) ]
italia_morti=[ sum(row[i] for row in morti_lista) for i in range(len(morti_lista[0])) ]
italia_dtot_morti=[ sum(row[i] for row in dtot_morti_lista) for i in range(len(dtot_morti_lista[0])) ]
italia_tamponi=[ sum(row[i] for row in tamponi_lista) for i in range(len(tamponi_lista[0])) ]
italia_dtot_tamponi=[ sum(row[i] for row in dtot_tamponi_lista) for i in range(len(dtot_tamponi_lista[0])) ]
italia_ospedalizzati=[ sum(row[i] for row in ospedalizzati_lista) for i in range(len(ospedalizzati_lista[0])) ]
italia_dtot_ospedalizzati=[ sum(row[i] for row in dtot_ospedalizzati_lista) for i in range(len(dtot_ospedalizzati_lista[0])) ]
italia_intensiva=[ sum(row[i] for row in intensiva_lista) for i in range(len(intensiva_lista[0])) ]
italia_dtot_intensiva=[ sum(row[i] for row in dtot_intensiva_lista) for i in range(len(dtot_intensiva_lista[0])) ]
italia_ricoverati=[ sum(row[i] for row in ricoverati_lista) for i in range(len(ricoverati_lista[0])) ]
italia_dtot_ricoverati=[ sum(row[i] for row in dtot_ricoverati_lista) for i in range(len(dtot_ricoverati_lista[0])) ]

italia_PLricov=np.sum(datiPLricov)
italia_PLti=np.sum(datiPLti)
italia_PLtiDL=np.sum(datiPLtiDL)                        # correggo secondo stima da DL

#popolazione regionale
pop_lista=[]
for key in diz:
    posizione=int(np.where(datipop[:,0]==key)[0])
    pop_lista.append(datipop[posizione,1])

pop_italia = int(59257566)           #http://dati.istat.it/Index.aspx?DataSetCode=DCIS_POPRES1

# incidenza
italia_casi_sett=np.array(italia_tot)[mask_gio_succ]-np.array(italia_tot)[mask_gio]   #tecnicamente sommo i valori da venerdi compreso al giovedi successivo
italia_incidenza=100000*italia_casi_sett/pop_italia

italia_casi_sett_mobile=np.zeros(len(italia_tot)-7)                                 # media mobile
for j in range(7,len(italia_tot),1):
    italia_casi_sett_mobile[j-7]=np.array(italia_tot)[j] - np.array(italia_tot)[j-7]
italia_incidenza_mobile=100000*italia_casi_sett_mobile/pop_italia

#media mobile
passo=7
w = [1.0/passo]*passo


italia_der_norm=100000*np.array(italia_dtot)/pop_italia  # derivate normalizzate per popolazione
italia_der_norm_med=np.convolve(italia_der_norm,w,"valid") # media mobile

italia_der_morti_norm=100000*np.array(italia_dtot_morti)/pop_italia  # derivate normalizzate per popolazione
italia_der_morti_norm_med=np.convolve(italia_der_morti_norm,w,"valid") # media mobile


#preparo grafici      
label_lista=list(diz.values())
codici_lista=list(diz.keys())
regioni_plot=['Veneto','Lombardia']
indici_plot=[label_lista.index(regioni_plot[0]),label_lista.index(regioni_plot[1])]

NUM_COLORS = 21
cm = plt.get_cmap('tab20c')
colors=[]
for i in range(NUM_COLORS):
    colors.append(cm(1.*i/NUM_COLORS))  # color will now be an RGBA tuple

def diretta(x):
    return x*norm
def inversa(x):
    return x/norm



############ sandbox


############################# PLOT ############################################
#Italia
fig_italia=plt.figure(figsize=(15,10))
ax1=fig_italia.add_subplot(2,2,1)
ax1.set_title('Casi totali in Italia')
ax1.set_ylabel('Casi registrati')
ax1.xaxis.set_major_formatter(formatter)
ax1.plot_date(date_list,italia_tot, color='green', marker='None', linestyle='solid', label='Totali')      #marker='*'
ax1.plot_date(date_list,italia_attuali, color='black', marker='None', linestyle='solid', label='Attuali')    #,marker='+'
ax1.plot_date(date_list,10*np.array(italia_morti), color='red', marker='None', linestyle='solid', label='Morti (10:1)')
ax1.plot_date(date_list,100*np.array(italia_intensiva), color='cyan', marker='None', linestyle='solid', label='Intensiva (100:1)')
ax1.axhline(y=0, color='grey', linestyle='dotted')
ax1.set_xlim(date_list[371], date_list[N-1])

ax1.plot_date(date_list,np.array(italia_tamponi)/20., color='fuchsia', marker='None', linestyle='solid', alpha=0.3,label='Tamponi')  #tamponi
secaxy=ax1.secondary_yaxis('right',functions=(lambda x: x*20., lambda x: x/20.))
secaxy.set_ylabel('Tamponi')
secaxy.ticklabel_format(axis="y", style="sci", scilimits=(0,0))



ax2=fig_italia.add_subplot(2,2,2)
ax2.set_title('Nuovi casi giornalieri in Italia')
ax2.set_ylabel('Nuovi casi giornalieri')
ax2.xaxis.set_major_formatter(formatter)
ax2.plot_date(date_list,italia_dtot, color='green',  marker='None', linestyle='solid', label='Totali')       #totali
ax2.plot_date(date_list,italia_dtot_attuali, color='black',  marker='None', linestyle='solid',label='Attuali')  #attualmente positivi
ax2.axhline(y=0, color='grey', linestyle='dotted')
ax2.set_ylim(-5000, 300000) 
ax2.set_xlim(date_list[371], date_list[N-1])


ax2.plot_date(date_list,np.array(italia_dtot_tamponi)/10., color='fuchsia', marker='None', linestyle='solid',alpha=0.3, label='Tamponi')  #tamponi
secaxy=ax2.secondary_yaxis('right',functions=(lambda x: x*10., lambda x: x/10.))
secaxy.set_ylabel('Tamponi giornalieri')
secaxy.ticklabel_format(axis="y", style="sci", scilimits=(0,0))


ax3=fig_italia.add_subplot(2,2,3)
ax3.set_title('Tasso di positività in Italia')
ax3.set_ylabel('Percentuale positivi (%)')
ax3.xaxis.set_major_formatter(formatter)
ax3.plot_date(date_list,100*np.asarray(italia_dtot)/np.asarray(italia_dtot_tamponi),color='dodgerblue', marker='None', linestyle='solid', label='% positivi')  #marker='.',
ax3.axhline(y=0, color='grey', linestyle='dotted')
ax3.set_ylim(-1, 1)
ax3.set_xlim(date_list[371], date_list[N-1])


ax4=fig_italia.add_subplot(2,2,4)
ax4.set_title('Nuovi casi in intensiva e morti giornalieri in Italia')
ax4.set_ylabel('Nuovi morti giornalieri')
ax4.xaxis.set_major_formatter(formatter)
ax4.plot_date(date_list,italia_dtot_morti, color='red', marker='None', linestyle='solid', label='Morti')       #morti
#ax4.plot_date(date_list,italia_dtot_intensiva, color='cyan', marker='None', linestyle='solid', label='Intensiva')       #intensiva
ax4.axhline(y=0, color='grey', linestyle='dotted')
ax4.set_xlim(date_list[371], date_list[N-1])
ax4.set_ylim(-400, 700)

##norm=np.amax(italia_dtot_intensiva)/np.amax(italia_dtot_morti)
##ax4.plot_date(date_list,italia_dtot_intensiva/norm, color='cyan', marker='None', linestyle='solid',label='Intensive')  #intensiva
##secaxy=ax4.secondary_yaxis('right',functions=(diretta,inversa))
##secaxy.set_ylabel('Intensive giornaliere')

ax4.plot_date(date_list,np.array(italia_dtot_intensiva)*4., color='cyan', marker='None', linestyle='solid',label='Intensive')  #intensiva
secaxy=ax4.secondary_yaxis('right',functions=(lambda x: x/4., lambda x: x*4.))
secaxy.set_ylabel('Intensive giornaliere')

ax3.set_ylim(0,35)
ax1.grid(linestyle='dotted')
ax2.grid(linestyle='dotted')
ax3.grid(linestyle='dotted')
ax4.grid(linestyle='dotted')

ax1.legend(loc=2)
ax2.legend(loc=2)
##ax3.legend(loc=2)
ax4.legend(loc=4)
plt.tight_layout()
plt.show()
fig_italia.savefig('italia.pdf')






#regioni derivate
fig_regionider, axs=plt.subplots(nrows=4, ncols=5, figsize=(15,10), sharex=True)
fig_regionider.suptitle('Nuovi casi giornalieri negli ultimi due mesi')
for i,ax in zip(np.arange(0,20),axs.flat):
    ax.set_title(label=label_lista[i])
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(linestyle='dotted')
                                                              
    ax.plot_date(date_list,dtot_lista[i], color='green',  marker='None', linestyle='solid')       #totali
    #ax.plot_date(date_list,dtot_morti_lista[i], color='red', marker='None', linestyle='solid')       #morti 
    ax.axhline(y=0, color='grey', linestyle='dotted')

##    #norm=np.amax(dtot_tamponi_lista[i][-40:])/(np.amax(dtot_lista[i][-40:]))
##    ax.plot_date(date_list,np.array(dtot_tamponi_lista[i])/10., color='fuchsia', marker='None', linestyle='solid', alpha=0.3)  #tamponi
##    secaxy=ax.secondary_yaxis('right',functions=(lambda x: x*10., lambda x: x/10.))
##    secaxy.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

    ax.set_xlim(date_list[N-60], date_list[N-1])
    ax.set_ylim(0, np.amax(dtot_lista[i][-60:] + int(5)))
    ax.set_xticks([date_list[N-60], date_list[N-30], date_list[N-1]])
  
plt.tight_layout()
fig_regionider.subplots_adjust(top=0.91)
plt.show()
fig_regionider.savefig('regioni_der.pdf')







# nuovi casi normalizzati per popolazione (media mobile)
fig_italia=plt.figure(figsize=(15,10))
ax=fig_italia.add_subplot(1,1,1)
ax.set_title('Nuovi casi giornalieri ogni 100 000 abitanti regionali negli ultimi due mesi  (media mobile settimanale)')
ax.set_ylabel('Nuovi casi')
ax.xaxis.set_major_formatter(formatter)
ax.set_xlim(date_list[N-60], date_list[N-1])
#ax.set_ylim(0, 100000*np.amax(dtot_lista[i][-40:])/pop_lista[i])
ax.set_xticks([date_list[N-60], date_list[N-30], date_list[N-1]])
ax.set_ylim(0,600)
ax.axhline(y=0, color='grey', linestyle='dotted')

for i in range(Nregioni):
    der_norm=100000*dtot_lista[i]/pop_lista[i]  # derivate normalizzate per popolazione
    der_norm_med=np.convolve(der_norm,w,"valid") # media mobile
    if der_norm_med[N-7] >= italia_der_norm_med[N-7]:    #1.7
        ax.plot_date(date_list[-len(der_norm_med):],der_norm_med, color=colors[i], marker='None', linestyle='solid', label=label_lista[i])
    else:
        ax.plot_date(date_list[-len(der_norm_med):],der_norm_med, color=colors[i], marker='None', linestyle='dotted', alpha=0.2, label=label_lista[i])

ax.plot_date(date_list[-len(der_norm_med):],italia_der_norm_med, color='black', marker='None', linestyle='dashed',  label='Nazionale')

plt.grid(linestyle='dotted')
ax.legend(loc=2)
plt.tight_layout()
plt.show()
fig_italia.savefig('regioni_der_pop_NORM.pdf')





#regioni derivate intensive e morti
fig_regioniderint, axs=plt.subplots(nrows=4, ncols=5, figsize=(15,10), sharex=True, sharey=True)
fig_regioniderint.suptitle('Nuovi casi in intensiva (azzurro) e morti (rosso) negli ultimi due mesi')
for i,ax in zip(np.arange(0,20),axs.flat):
    ax.set_title(label=label_lista[i])
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(linestyle='dotted')
                                                              
    #ax.plot_date(date_list,dtot_ospedalizzati_lista[i], color='grey',  marker='None', linestyle='solid')       #ospedalizzati
    #ax.plot_date(date_list,np.array(intensiva_lista[i])/3., color='dodgerblue', marker='None', linestyle='dotted', label='Intensiva (100:1)')  #TOTALI intensive
    ax.plot_date(date_list,dtot_intensiva_lista[i], color='cyan',  marker='None', linestyle='solid')       #intensive
    ax.plot_date(date_list,dtot_morti_lista[i], color='red', marker='None', linestyle='solid')       #morti 
    ax.axhline(y=0, color='grey', linestyle='dotted')

    ax.set_xlim(date_list[N-60], date_list[N-1])
    #ax.set_ylim(0, np.amax(dtot_intensiva_lista[i][-40:]))
    ax.set_ylim(-20,150)
    ax.set_xticks([date_list[N-60], date_list[N-30], date_list[N-1]])
  
plt.tight_layout()
fig_regioniderint.subplots_adjust(top=0.91)
plt.show()
fig_regioniderint.savefig('regioni_der_int.pdf')





# nuovi morti normalizzati per popolazione (media mobile)
fig_italia=plt.figure(figsize=(15,10))
ax=fig_italia.add_subplot(1,1,1)
ax.set_title('Nuovi morti giornalieri ogni 100 000 abitanti regionali negli ultimi due mesi (media mobile settimanale)')
ax.set_ylabel('Nuovi morti')
ax.xaxis.set_major_formatter(formatter)
ax.set_xlim(date_list[N-60], date_list[N-1])
#ax.set_ylim(0, 100000*np.amax(dtot_lista[i][-40:])/pop_lista[i])
ax.set_xticks([date_list[N-60], date_list[N-30], date_list[N-1]])
ax.set_ylim(0.,1.)
ax.axhline(y=0, color='grey', linestyle='dotted')

for i in range(Nregioni):
    der_norm=100000*dtot_morti_lista[i]/pop_lista[i]  # derivate normalizzate per popolazione
    der_norm_med=np.convolve(der_norm,w,"valid") # media mobile
    if der_norm_med[N-7] >= italia_der_morti_norm_med[N-7]:   #0.05
        ax.plot_date(date_list[-len(der_norm_med):],der_norm_med, color=colors[i], marker='None', linestyle='solid', label=label_lista[i])
    else:
        ax.plot_date(date_list[-len(der_norm_med):],der_norm_med, color=colors[i], marker='None', linestyle='dotted', alpha=0.2, label=label_lista[i])

ax.plot_date(date_list[-len(der_norm_med):],italia_der_morti_norm_med, color='black', marker='None', linestyle='dashed',  label='Nazionale')


plt.grid(linestyle='dotted')
ax.legend(loc=2)
plt.tight_layout()
plt.show()
fig_italia.savefig('regioni_morti_der_pop_NORM.pdf')


###percentuale italia -> stackplot (media mobile)
##fig_stackplotM, ax=plt.subplots(figsize=(15,10))
##ax.set_title('Incidenza nuovi positivi giornalieri regionali sui nazionali (dati della media mobile)')
##ax.set_ylabel('Percentuale')
##ax.xaxis.set_major_formatter(formatter)
##
##valori=[]
##for i in range(0,21):
##    med_dtot_reg=np.convolve(dtot_lista[i],w,"valid")       #media mobile dati regionali
##    med_dtot_italia=np.convolve(italia_dtot,w,"valid")       #media mobile dati nazionali
##    med_percentuale=(100*med_dtot_reg/med_dtot_italia).astype(float)
##    valori.append(med_percentuale)
##y=np.vstack(valori)
##ax.stackplot(date_list[1:len(med_percentuale)+1],y,colors=colors, labels=label_lista)
##
##plt.grid(linestyle='dotted')
##plt.ylim(top=100, bottom=0)
##plt.xlim(left=date_list[N-365], right=date_list[len(med_percentuale)])
##
##handles, labels = ax.get_legend_handles_labels()
##ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0009,0.5))   
##plt.tight_layout()
##fig_stackplotM.subplots_adjust(top=0.96, right=0.87)
##plt.show()
##fig_stackplotM.savefig('italia_stackplot_mobile.pdf')
##
##
##
##
##
###percentuale italia -> stackplot morti (media mobile)
##fig_stackplotmortiM, ax=plt.subplots(figsize=(15,10))
##ax.set_title('Incidenza nuovi morti giornalieri regionali sui nazionali (dati della media mobile)')
##ax.set_ylabel('Percentuale')
##ax.xaxis.set_major_formatter(formatter)
##
##valori=[]
##for i in range(0,21):
##    med_dtot_reg=np.convolve(dtot_morti_lista[i],w,"valid")       #media mobile dati regionali
##    med_dtot_italia=np.convolve(italia_dtot_morti,w,"valid")       #media mobile dati nazionali
##    med_percentuale=(100*med_dtot_reg/med_dtot_italia).astype(float)
##    valori.append(med_percentuale)
##y=np.vstack(valori)
##ax.stackplot(date_list[1:len(med_percentuale)+1],y,colors=colors, labels=label_lista)
##
##plt.grid(linestyle='dotted')
##plt.ylim(top=100, bottom=0)
##plt.xlim(left=date_list[N-365], right=date_list[len(med_percentuale)])
##
##handles, labels = ax.get_legend_handles_labels()
##ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0009,0.5))   
##plt.tight_layout()
##fig_stackplotmortiM.subplots_adjust(top=0.96, right=0.87)
##plt.show()
##fig_stackplotmortiM.savefig('italia_stackplot_mobile_morti.pdf')




# saturazione intensive e ricoveri
fig_saturazione=plt.figure(figsize=(15,10))
ax=fig_saturazione.add_subplot(1,1,1)
ax.set_title('Saturazione posti letto in terapia intensiva e ricoveri il giorno '+ str(date_list[N-1].strftime('%d/%m/%y')) + ' con incidenza settimanale aggiornata al '+ str(date_list[N-1].strftime('%d/%m/%y')) + ' e storico dei '+str(7) +' giorni precedenti' +
             '\n \n NB: STIME dei posti letto disponibili in terapia intensiva secondo il report ISS calcolate da Vittorio Nicoletta al '+ str(report) +
             '\n Fonte: https://docs.google.com/spreadsheets/d/e/2PACX-1vSwS7SWNpcv8qy1wZHK7DvWWX7RgDEYzYjzAm_TdExqax3waIuN2Bd0rj2OAh9YRJSDX-QbnqEkfdqJ/pubhtml#' +
             '\n \n NB: Posti letto disponibili in area non critica secondo Agenas https://www.agenas.gov.it/covid19/web/index.php?r=site%2Ftab2 aggiornati al ' + str(agenas))
ax.set_xlabel('Percentuale occupazione posti letto in area non critica')
ymax_colori=15
ax.set_ylim(0,ymax_colori)
xmax=35.
ax.set_xlim(0,xmax)
ax.axvline(x=15, color='k', linestyle='dotted') 
ax.axvline(x=30, color='k', linestyle='dotted')
ax.axvline(x=40, color='k', linestyle='dotted') 
ax.axhline(y=10, color='k', linestyle='dotted')
ax.axhline(y=20, color='k', linestyle='dotted')
ax.axhline(y=30, color='k', linestyle='dotted')
ax.axhspan(10, 20, 30./xmax ,1, facecolor='yellow',alpha=0.3)
ax.axhspan(10, ymax_colori, 15./xmax ,30./xmax, facecolor='yellow',alpha=0.3)
ax.axhspan(20, 30, 40./xmax ,1, facecolor='orange',alpha=0.3)
ax.axhspan(20, ymax_colori, 30./xmax ,40./xmax, facecolor='orange',alpha=0.3)
ax.axhspan(30, ymax_colori, 40./xmax ,1, facecolor='red',alpha=0.3)
ax.axhspan(30, ymax_colori, 40./xmax ,40./xmax, facecolor='red',alpha=0.3)

italia_ricov=100*(italia_ricoverati/italia_PLricov)
italia_int=100*(italia_intensiva/italia_PLtiDL)


for i in range(Nregioni):
    reg_casi_sett_mobile=np.zeros(len(totali_lista[i])-7)                                 # media mobile
    for j in range(7,len(totali_lista[i]),1):
        reg_casi_sett_mobile[j-7]=np.array(totali_lista[i])[j] - np.array(totali_lista[i])[j-7]
    reg_incidenza_mobile=100000*reg_casi_sett_mobile/pop_lista[i]                       #incidenza ogni 100 000 abitanti regionali (dato attuale)

    reg_int=100*(intensiva_lista[i,N-1]/datiPLtiDL[i])  # percentuale intensive rispetto ai PL occupabili, dato odierno corretto per posti da DL
    reg_ricov=100*(ricoverati_lista[i,N-1]/datiPLricov[i])  # percentuale ricoveri rispetto ai PL occupabili, dato odierno

    reg_int7=100*(intensiva_lista[i,-7:]/datiPLtiDL[i])         # dati degli ultimi 7 giorni
    reg_ricov7=100*(ricoverati_lista[i,-7:]/datiPLricov[i])     # dati degli ultimi 7 giorni

    if int (150) > reg_incidenza_mobile[-1] >= int(50):
        ax.scatter(reg_ricov, reg_int, color='dodgerblue',  label=label_lista[i]+' : '+ str(round(reg_incidenza_mobile[-1],2)))
        ax.annotate(label_lista[i], (reg_ricov, reg_int), xytext=(reg_ricov+0.05, reg_int+0.1),color='dodgerblue')
        ax.plot(reg_ricov7, reg_int7, linestyle='solid', alpha=0.5, color='dodgerblue')
            
    elif int(1000) > reg_incidenza_mobile[-1] >= int(150):
        ax.scatter(reg_ricov, reg_int, color='red',  label=label_lista[i]+' : '+ str(round(reg_incidenza_mobile[-1],2)))
        ax.annotate(label_lista[i], (reg_ricov, reg_int), xytext=(reg_ricov+0.05, reg_int+0.1), color='red')
        ax.plot(reg_ricov7, reg_int7, linestyle='solid', alpha=0.5, color='red')

    elif reg_incidenza_mobile[-1] >= int(1000):
        ax.scatter(reg_ricov, reg_int, color='darkred',  label=label_lista[i]+' : '+ str(round(reg_incidenza_mobile[-1],2)))
        ax.annotate(label_lista[i], (reg_ricov, reg_int), xytext=(reg_ricov+0.05, reg_int+0.1), color='darkred')
        ax.plot(reg_ricov7, reg_int7, linestyle='solid', alpha=0.5, color='darkred')
            
    else:
        ax.scatter(reg_ricov, reg_int, color='grey', label=label_lista[i]+' : '+ str(round(reg_incidenza_mobile[-1],2)))
        ax.annotate(label_lista[i], (reg_ricov, reg_int), xytext=(reg_ricov+0.05, reg_int+0.1), color='grey')
        ax.plot(reg_ricov7, reg_int7, linestyle='solid', alpha=0.5, color='grey')
    


ax.scatter(italia_ricov[N-1],italia_int[N-1], color='black', marker='D', label='Nazionale'+' : '+ str(round(italia_incidenza_mobile[-1],2)))   
mplcursors.cursor()    # hover=True or no argument to click

leg=ax.legend(title='Incidenza al '+ str(date_list[N-1].strftime('%d/%m/%y')) + '\n ', loc=2)                         # va messo PRIMA di settare i colori!!
for h, t in zip(leg.legendHandles, leg.get_texts()):
    t.set_color(h.get_facecolor()[0])


plt.grid(linestyle='dotted')
plt.tight_layout()
plt.show()
fig_saturazione.savefig('saturazione.pdf')





# saturazione intensive e ricoveri secondo stima al 20/8/2021 https://twitter.com/vi__enne/status/1428732053512671233
fig_saturazione=plt.figure(figsize=(15,10))
ax=fig_saturazione.add_subplot(1,1,1)
ax.set_title('Saturazione posti letto in terapia intensiva e ricoveri il giorno giovedì '+ str(date_list[mask_gio_succ[-1]].strftime('%d/%m/%y')) + ' con incidenza aggiornata a giovedì '+ str(date_list[mask_gio_succ[-1]].strftime('%d/%m/%y')) +
             '\n \n NB: STIME dei posti letto disponibili in terapia intensiva secondo il report ISS calcolate da Vittorio Nicoletta al '+ str(report) +
             '\n Fonte: https://docs.google.com/spreadsheets/d/e/2PACX-1vSwS7SWNpcv8qy1wZHK7DvWWX7RgDEYzYjzAm_TdExqax3waIuN2Bd0rj2OAh9YRJSDX-QbnqEkfdqJ/pubhtml#' +
             '\n \n NB: Posti letto disponibili in area non critica secondo Agenas https://www.agenas.gov.it/covid19/web/index.php?r=site%2Ftab2 aggiornati al ' + str(agenas))
ax.set_ylabel('Percentuale occupazione posti letto in terapia intensiva')
ax.set_xlabel('Percentuale occupazione posti letto in area non critica')
ax.set_ylim(0,ymax_colori)
ax.set_xlim(0,xmax)
ax.axvline(x=15, color='k', linestyle='dotted') 
ax.axvline(x=30, color='k', linestyle='dotted')
ax.axvline(x=40, color='k', linestyle='dotted') 
ax.axhline(y=10, color='k', linestyle='dotted')
ax.axhline(y=20, color='k', linestyle='dotted')
ax.axhline(y=30, color='k', linestyle='dotted')
ax.axhspan(10, 20, 30./xmax ,1, facecolor='yellow',alpha=0.3)
ax.axhspan(10, ymax_colori, 15./xmax ,30./xmax, facecolor='yellow',alpha=0.3)
ax.axhspan(20, 30, 40./xmax ,1, facecolor='orange',alpha=0.3)
ax.axhspan(20, ymax_colori, 30./xmax ,40./xmax, facecolor='orange',alpha=0.3)
ax.axhspan(30, ymax_colori, 40./xmax ,1, facecolor='red',alpha=0.3)
ax.axhspan(30, ymax_colori, 40./xmax ,40./xmax, facecolor='red',alpha=0.3)


italia_ricov=100*(italia_ricoverati/italia_PLricov)
italia_int=100*(italia_intensiva/italia_PLtiDL)

for i in range(Nregioni):
    reg_casi_sett=np.array(totali_lista[i])[mask_gio_succ]-np.array(totali_lista[i])[mask_gio]   #tecnicamente sommo i valori da venerdi compreso al giovedi successivo
    reg_incidenza= 100000*reg_casi_sett/pop_lista[i]        #incidenza ogni 100 000 abitanti regionali (dato attuale del giovedi)

    reg_int=100*(intensiva_lista[i,mask_gio_succ[-1]]/datiPLtiDL[i])  # percentuale intensive rispetto ai PL occupabili, dato giovedi (stima secondo DL)
    reg_ricov=100*(ricoverati_lista[i,mask_gio_succ[-1]]/datiPLricov[i])  # percentuale ricoveri rispetto ai PL occupabili, dato giovedi

    if int (150) >reg_incidenza[-1] >= int(50):
        ax.scatter(reg_ricov, reg_int, color='dodgerblue',  label=label_lista[i]+' : '+ str(round(reg_incidenza[-1],2)))
        ax.annotate(label_lista[i], (reg_ricov, reg_int), xytext=(reg_ricov+0.05, reg_int+0.1),color='dodgerblue')
        #if (int(20) >= reg_int >= int(10)) & (int(30) >= reg_ricov >= int(15)):
        #    ax.annotate(label_lista[i]+' (G)', (reg_ricov, reg_int), xytext=(reg_ricov+0.05, reg_int+0.1),color='dodgerblue')
        #else:
            
    elif int(1000) > reg_incidenza[-1] >= int(150):
        ax.scatter(reg_ricov, reg_int, color='red',  label=label_lista[i]+' : '+ str(round(reg_incidenza[-1],2)))
        ax.annotate(label_lista[i], (reg_ricov, reg_int), xytext=(reg_ricov+0.05, reg_int+0.1), color='red')

    elif reg_incidenza[-1] >= int(1000):
        ax.scatter(reg_ricov, reg_int, color='darkred',  label=label_lista[i]+' : '+ str(round(reg_incidenza[-1],2)))
        ax.annotate(label_lista[i], (reg_ricov, reg_int), xytext=(reg_ricov+0.05, reg_int+0.1), color='darkred')            

    else:
        ax.scatter(reg_ricov, reg_int, color='grey', label=label_lista[i]+' : '+ str(round(reg_incidenza[-1],2)))
        ax.annotate(label_lista[i], (reg_ricov, reg_int), xytext=(reg_ricov+0.05, reg_int+0.1), color='grey')


ax.scatter(italia_ricov[mask_gio_succ[-1]],italia_int[mask_gio_succ[-1]], color='black', marker='D', label='Nazionale'+' : '+ str(round(italia_incidenza[-1],2)))   
mplcursors.cursor()    # hover=True or no argument to click

leg=ax.legend(title='Incidenza al '+ str(date_list[mask_gio_succ[-1]].strftime('%d/%m/%y')) + '\n ', loc=2)                         # va messo PRIMA di settare i colori!!
for h, t in zip(leg.legendHandles, leg.get_texts()):
    t.set_color(h.get_facecolor()[0])


plt.grid(linestyle='dotted')
plt.tight_layout()
plt.show()
fig_saturazione.savefig('saturazione_colori.pdf')






# ricoverati
fig_ricov=plt.figure(figsize=(15,10))
ax=fig_ricov.add_subplot(1,1,1)
ax.set_title('Occupazione posti letto in area non critica negli ultimi due mesi (dati Agenas)')
ax.set_ylabel('Percentuale occupazione posti letto in area non critica')
ax.xaxis.set_major_formatter(formatter)
ax.set_xlim(date_list[N-60], date_list[N-1])
#ax.set_ylim(0, 100000*np.amax(dtot_lista[i][-40:])/pop_lista[i])
ax.set_xticks([date_list[N-60], date_list[N-30], date_list[N-1]])
ax.set_ylim(0,xmax)
ax.axhline(y=0, color='grey', linestyle='dotted')
ax.axhline(y=15, color='k', linestyle='dotted')
ax.axhline(y=30, color='k', linestyle='dotted')
ax.axhline(y=40, color='k', linestyle='dotted')
ax.axhspan(15, 30, facecolor='yellow', alpha=0.1)
ax.axhspan(30, 40, facecolor='orange', alpha=0.1)
ax.axhspan(40, xmax, facecolor='red', alpha=0.1)

italia_ricov=100*(italia_ricoverati/italia_PLricov)
for i in range(Nregioni):
    reg_ricov=100*(ricoverati_lista[i]/datiPLricov[i])  # percentuale ricoveri rispetto ai PL occupabili
    if reg_ricov[N-1] >= italia_ricov[N-1]:   
        ax.plot_date(date_list[N-60:N],reg_ricov[N-60:N], color=colors[i], marker='None', linestyle='solid', label=label_lista[i])
    else:
        ax.plot_date(date_list[N-60:N],reg_ricov[N-60:N], color=colors[i], marker='None', linestyle='dotted', alpha=0.2, label=label_lista[i])

ax.plot_date(date_list[N-60:N],italia_ricov[N-60:N], color='black', marker='None', linestyle='dashed',  label='Nazionale')

plt.grid(linestyle='dotted')
ax.legend(loc=2)
plt.tight_layout()
plt.show()
fig_ricov.savefig('ricoverati.pdf')




# intensive
fig_int=plt.figure(figsize=(15,10))
ax=fig_int.add_subplot(1,1,1)
ax.set_title('Occupazione posti letto in terapia intensiva negli ultimi due mesi (Stime aggiornate al ' + str(report) + ')')
ax.set_ylabel('Percentuale occupazione posti letto in terapia intensiva')
ax.xaxis.set_major_formatter(formatter)
ax.set_xlim(date_list[N-60], date_list[N-1])
#ax.set_ylim(0, 100000*np.amax(dtot_lista[i][-40:])/pop_lista[i])
ax.set_xticks([date_list[N-60], date_list[N-30], date_list[N-1]])
ax.set_ylim(0.,ymax_colori)
ax.axhline(y=0, color='grey', linestyle='dotted')
ax.axhline(y=10, color='k', linestyle='dotted')
ax.axhline(y=20, color='k', linestyle='dotted')
ax.axhline(y=30, color='k', linestyle='dotted')
ax.axhspan(10, 20, facecolor='yellow', alpha=0.1)
ax.axhspan(20, 30, facecolor='orange', alpha=0.1)
ax.axhspan(30, ymax_colori, facecolor='red', alpha=0.1)

italia_int=100*(italia_intensiva/italia_PLtiDL)
for i in range(Nregioni):
    reg_int=100*(intensiva_lista[i]/datiPLtiDL[i])  # percentuale intensive rispetto ai PL occupabili
    if reg_int[N-1] >= italia_int[N-1]:   
        ax.plot_date(date_list[N-60:N],reg_int[N-60:N], color=colors[i], marker='None', linestyle='solid', label=label_lista[i])
    else:
        ax.plot_date(date_list[N-60:N],reg_int[N-60:N], color=colors[i], marker='None', linestyle='dotted', alpha=0.2, label=label_lista[i])

ax.plot_date(date_list[N-60:N],italia_int[N-60:N], color='black', marker='None', linestyle='dashed',  label='Nazionale')


plt.grid(linestyle='dotted')
ax.legend(loc=2)
plt.tight_layout()
plt.show()
fig_int.savefig('intensive.pdf')






# incidenza
fig_incidenza=plt.figure(figsize=(15,10))
ax=fig_incidenza.add_subplot(1,1,1)
ax.set_title('Incidenza per 100 000 abitanti negli ultimi due mesi. Dato valutato settimanalmente ogni giovedì')
ax.set_ylabel('Incidenza per 100 000 abitanti')
ax.xaxis.set_major_formatter(formatter)
ax.set_xlim(date_list[mask_gio_succ[-9]], date_list[mask_gio_succ[-1]])
#ax.set_ylim(0, 100000*np.amax(dtot_lista[i][-40:])/pop_lista[i])
ax.set_xticks(np.array(date_list)[mask_gio_succ[-9:]])
ymax=3500
ax.set_ylim(0.,ymax)
ax.axhline(y=0, color='grey', linestyle='dotted')
ax.axhline(y=50, color='black', linestyle='dotted')
ax.axhline(y=150, color='black', linestyle='dotted')
ax.axhspan(50, 150, facecolor='yellow', alpha=0.1)
ax.axhspan(150, ymax, facecolor='orange', alpha=0.1)


for i in range(Nregioni):
    reg_casi_sett=np.array(totali_lista[i])[mask_gio_succ]-np.array(totali_lista[i])[mask_gio]   #tecnicamente sommo i valori da venerdi compreso al giovedi successivo
    reg_incidenza= 100000*reg_casi_sett/pop_lista[i]        #incidenza ogni 100 000 abitanti regionali (dato attuale)

    if reg_incidenza[-1] >= italia_incidenza[-1]:
        ax.plot_date(np.array(date_list)[mask_gio_succ[9:]],reg_incidenza[9:], color=colors[i], marker='None', linestyle='solid', label=label_lista[i])
    else:
        ax.plot_date(np.array(date_list)[mask_gio_succ[9:]],reg_incidenza[9:], color=colors[i], marker='None', linestyle='dotted', alpha=0.2, label=label_lista[i])

ax.plot_date(np.array(date_list)[mask_gio_succ[9:]],italia_incidenza[9:], color='black', marker='None', linestyle='dashed',  label='Nazionale')

plt.grid(linestyle='dotted')
ax.legend(loc=2)
plt.tight_layout()
plt.show()
fig_incidenza.savefig('incidenza.pdf')

