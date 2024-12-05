from mtcnn import MTCNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import array_to_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import tensorflow as tf
import os
from sklearn.utils import shuffle

import os
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

def rensa_skärm():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')

class Modell:
    def __init__(self, bildstorlek_höjd=224, bildstorlek_bredd=224):
        
        self.bildstorlek_höjd = bildstorlek_höjd
        self.bildstorlek_bredd = bildstorlek_bredd
        #self.bildmapp_källa = "" #bildmapp_källa
        #self.bildmapp_ansikten = ""#bildmapp_ansikten
        self.kategorier = []

        self.antal_kanaler = 3 # 1 = gråskala 3 = rgb
        self.detektor = MTCNN()

        self.model = keras.models
        self.modell_redo = False
        self.x_train = np.array
        self.x_test = np.array
        self.y_train = np.array
        self.y_test = np.array
        self.x_förutsägelse = np.array
        self.y_förutsägelse = np.array
        self.bildfil = ""
 
        self.bildbehandling_statistik_antal_filer = 0
        self.bildbehandling_statistik_antal_ansiktsfiler = 0
        self.bildbehandling_statistik_antal_ansiktsfiler_suddiga = 0
        self.bildbehandling_statistik_antal_ansiktsfiler_lågupplösning = 0
        
    def __str__(self):
        if self.modell_redo == True:
            return("Redo")
        else:
            return("Ej redo. Ladda eller träna modell först")

    def nollställ_bildbehandling_statistik(self):
        
        self.bildbehandling_statistik_antal_filer = 0
        self.bildbehandling_statistik_antal_ansiktsfiler = 0
        self.bildbehandling_statistik_antal_ansiktsfiler_suddiga = 0
        self.bildbehandling_statistik_antal_ansiktsfiler_lågupplösning = 0

    def behandla_bild(self, bild):
        # ändrar storlek till vår definerade standard (224x224)
        bild = cv2.resize(bild, (self.bildstorlek_bredd,self.bildstorlek_höjd))
        return(bild)

    def normalisera(self):
        
        # normalisera bilder enligt definerad storlek (224x224), antal kanaler
        self.x_train = self.x_train.reshape(-1, self.bildstorlek_höjd, self.bildstorlek_bredd, self.antal_kanaler) 
        self.x_test = self.x_test.reshape(-1, self.bildstorlek_höjd, self.bildstorlek_bredd, self.antal_kanaler)

        # reducera pixlar till värde mellan 0..1
        self.x_train = self.x_train / 255.0 
        self.x_test = self.x_test / 255.0        

        # encoda kategorier
        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)


    def ladda_data_förutsägelse_mapp(self, mapp):
        
        # denna procedur används för att ladda en ansiktsbild (ett ansikte) per bild, går igenom en mapp
        x_förutsägelse = []
        y_förutsägelse = []
        bild_fil_nr = 0

        # iterera  mapp
        for bild_fil in os.listdir(mapp):
            print(f"bild {bild_fil}")
            try:
                bild_fil_nr += 1
                bild = cv2.imread(os.path.join(mapp, bild_fil))
                bild = self.behandla_bild(bild)
                x_förutsägelse.append(bild)
                y_förutsägelse.append(bild_fil)
            except Exception as e:
                print(f"Ett fel uppstod: {e}")   

        self.x_förutsägelse = np.array(x_förutsägelse)
        self.y_förutsägelse = np.array(y_förutsägelse)       

        self.x_förutsägelse = self.x_förutsägelse.reshape(-1, self.bildstorlek_höjd, self.bildstorlek_bredd, self.antal_kanaler) 
        self.x_förutsägelse = self.x_förutsägelse / 255.0 


    def ladda_data_förutsägelse(self, bild_fil, bilder):
        
        # denna procedur används för att ladda en eller flera ansikten i en eller flera ansiktsbild(er)
        x_förutsägelse = []
        y_förutsägelse = []
        bild_fil_nr = 0

        for bild in bilder:
            try:
                bild_fil_nr += 1
                bild = self.behandla_bild(bild)
                x_förutsägelse.append(bild)
                y_förutsägelse.append(bild_fil)
            except Exception as e:
                print(f"Ett fel uppstod: {e}")   

        self.x_förutsägelse = np.array(x_förutsägelse)
        self.y_förutsägelse = np.array(y_förutsägelse)       

        self.x_förutsägelse = self.x_förutsägelse.reshape(-1, self.bildstorlek_höjd, self.bildstorlek_bredd, self.antal_kanaler) 
        self.x_förutsägelse = self.x_förutsägelse / 255.0 


    def ladda_data(self, bildmapp_ansikten, andel_träning=0.8, andel_test=0.2, blanda=True, skippa_augmenterad_data=True):

        # ladda förbehandlade bilder för träning/test enligt given ratio, standard träning 80% test 20%
        x_train = []
        y_train = []
        x_test = []
        y_test = []       

        # varje undermapp är en kategori = namn på person
        self.kategorier = [i for i in os.listdir(bildmapp_ansikten) if os.path.isdir(os.path.join(bildmapp_ansikten, i))]

        # iterera alla kategorier (undermappar i ansiktsmapp)
        for kategori in self.kategorier:
            mapp = os.path.join(bildmapp_ansikten, kategori)
            mapp_id = self.kategorier.index(kategori)             
            if skippa_augmenterad_data==True:   # om vi inte vill ha med augmenterade bilder exkluderas dessa då icke augmenterade bilder innehåller magiskt token _A_
                antal_filer = len([f for f in os.listdir(mapp) if os.path.isfile(os.path.join(mapp, f)) and f.endswith(".jpg") and f.find("_A_") > -1])    
            else:   # alla bilder
                antal_filer = len([f for f in os.listdir(mapp) if os.path.isfile(os.path.join(mapp, f)) and f.endswith(".jpg")])
            # räkna ut antal träning/test bilder
            antal_filer_träning = int(andel_träning * antal_filer)
            antal_filer_test = antal_filer - antal_filer_träning
            print(f"Kategori: {kategori} antal träningsfiler {antal_filer_träning} antal testfiler {antal_filer_test}.")
            bild_fil_nr = 0
            # itereration av bildfiler i mapp
            for bild_fil in os.listdir(mapp):
                try:
                    if ((skippa_augmenterad_data==True and bild_fil.find("_A_") > -1) or skippa_augmenterad_data==False):
                        bild_fil_nr += 1
                        bild = cv2.imread(os.path.join(mapp, bild_fil))
                        bild = self.behandla_bild(bild) # tillse att bilden är i rätt storlek

                        if bild_fil_nr <= antal_filer_träning:
                            x_train.append(bild)
                            y_train.append(mapp_id)
                        else:
                            x_test.append(bild)
                            y_test.append(mapp_id)
                except Exception as e:
                    print(f"Ett fel uppstod: {e}")   

        # gör om till numpy array
        self.x_train = np.array(x_train)
        self.x_test = np.array(x_test)
        self.y_train = np.array(y_train)
        self.y_test = np.array(y_test)       

        # om blandning önskas, använd shuffle
        if blanda==True:
            self.x_train, self.y_train = shuffle(self.x_train, self.y_train)
            self.x_test, self.y_test = shuffle(self.x_test, self.y_test)

    def skapa_modell(self, antal_filter=32, dropout=0.5):

        # skapar en sekventiell CNN keras modell
        # inspiration: https://learnopencv.com/implementing-cnn-tensorflow-keras/
        # som input anger vi storlek/shape på våra bilder
        # för att undvika overfitting används dropout 0.25 - 0.5
        # softmax räknar ut andel sannolikhet per kategori
        self.model = keras.Sequential()
        self.model.add(keras.Input(shape=(self.bildstorlek_höjd, self.bildstorlek_bredd, self.antal_kanaler)))
        self.model.add(keras.layers.Conv2D(antal_filter, (3,3), activation="relu"))        
        self.model.add(keras.layers.MaxPooling2D((2,2)))
        self.model.add(keras.layers.Conv2D(antal_filter, (3,3), activation="relu"))        
        self.model.add(keras.layers.MaxPooling2D((2,2)))
        self.model.add(keras.layers.Flatten())        
        self.model.add(keras.layers.Dense(128, activation="relu"))        
        self.model.add(keras.layers.Dropout(dropout))
        self.model.add(keras.layers.Dense(len(self.kategorier), activation='softmax'))


    def träna(self, antal_epochs=20, batch_storlek=32):

        #self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['CategoricalAccuracy'])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # callback funktion för att få ut test noggrannhet per kategori
        noggrannhet = Noggranhet((self.x_test, self.y_test), self.kategorier)

        # gör fit på modell på träning/test data och kör definerade antalet epoker, skicka in callback funktion för att få statistik på noggrannhet per kategori
        # träningsdata delas upp i angivet antal batcher
        self.statistik = self.model.fit(self.x_train, self.y_train, epochs=antal_epochs, shuffle=True, verbose=1,validation_data=(self.x_test, self.y_test),callbacks=[noggrannhet], batch_size=batch_storlek)

        # sätt flagga att modell är redo att användas        
        self.modell_redo = True
        # print(self.model.summary())

        eval = self.model.evaluate(self.x_test, self.y_test, verbose=2)
        test_förlust, test_noggrannhet = self.model.evaluate(self.x_test, self.y_test)
        #test = self.model.evaluate(self.x_test, self.y_test)
        #print(self.statistik)
        print(f"\nTest tillförlitlighet: {test_noggrannhet * 100:.2f}%")

    def spara_modell(self, mapp_modell, filnamn="egnaprojektet"):

        # spara model till disk
        modell_filnamn = os.path.join(mapp_modell, filnamn) + ".keras"
        kategorier_filnamn = os.path.join(mapp_modell, filnamn) + ".txt"

        self.model.save(modell_filnamn, overwrite=True)

        # spara ner kategorier till textfil med samma namn
        with open(kategorier_filnamn, "w") as f:
            for kategori in self.kategorier:
                f.write('%s\n' %kategori)
        f.close()

    def ladda_modell(self, mapp_modell, filnamn="egnaprojektet"):

        # ladda modell från disk
        modell_filnamn = os.path.join(mapp_modell, filnamn) + ".keras"
        kategorier_filnamn = os.path.join(mapp_modell, filnamn) + ".txt"

        # läs in kategorier från textfil
        with open(kategorier_filnamn, "r") as f:
            for i in f.readlines():
                self.kategorier.append(i)

        self.model = keras.models.load_model(modell_filnamn, compile=True)
        self.modell_redo = True

    def förutspå_multipla_ansikten(self, bildfil, max_ansikten=5, min_pixels=2048):

        bild = cv2.imread(bildfil)
        bild_rgb = cv2.cvtColor(bild, cv2.COLOR_BGR2RGB)

        # spara bildfil namn
        self.bildfil = bildfil

        # använd MTCNN för att känna av ansikten i en bild
        ansikten = self.detektor.detect_faces(bild_rgb)
        ansikten_beskurna = []
        bild_h, bild_b, bild_kanaler = bild.shape   # hämta storlek på bild för att känna av bildens gränser

        # sortera ansikten med högst confidence först, om vi har fler bilder än max gräns så ta de mest sannolika att det är ansikte först
        ansikten_sorterad = sorted(ansikten, reverse=True, key=lambda face: face['confidence'])
        
        i = 0
        # iterera igenom ansikten och lägg till kvalificerde ansikten i en lista
        for ansikte in ansikten_sorterad:
            if i > max_ansikten:        # avbryt om vi överskrider max antal tillåtna ansikten
                break
            x, y, b, h = ansikte['box']
            upplösning = b * h
            if upplösning > min_pixels:     # om bilden uppfyller minimum krav på upplösning
                i += 1
                ansikte_beskuret = self.beskär_ansikte(bild, bild_h, bild_b, y, h, x, b, 0.2)
                ansikten_beskurna.append(ansikte_beskuret)
                
        self.ladda_data_förutsägelse(bildfil, ansikten_beskurna)
        self.förutspå()


    def förutspå(self):

        # gör model.predict på inladdade bilder som ska analyseras
        # visa resultat mga mathplotlib med bild och namn på person och träffsäkerhet. 
        # om träffsäkerheten är under 70% visas inget namn

        try:        

            if len(self.x_förutsägelse) == 0:
                raise Exception("Ingen data att förutse är laddad.")

            förutsägelse = self.model.predict(self.x_förutsägelse, verbose=0)

            for i in range(0, len(förutsägelse)):
                högst_sannolikhet_index = förutsägelse[i].argmax()  
                högst_sannolikhet = förutsägelse[i][högst_sannolikhet_index]
                if högst_sannolikhet > 0.7:
                    resultat = self.kategorier[högst_sannolikhet_index]
                    resultat_träffsäkerhet = f"({högst_sannolikhet * 100:.2f}%)"
                else:
                    resultat = ""
                    resultat_träffsäkerhet = ""

                plt.subplot(220 + (i+1))
                plt.title(f"Förutsägelse: {resultat} {resultat_träffsäkerhet}")
                plt.imshow(self.x_förutsägelse[i], cmap='gray')
                plt.grid(False)
                plt.axis('off')    
            
            plt.show()

        except Exception as e:
            print(f"Ett fel uppstod: {e}")   

    # funktion för att lägga till suffix resp prefix till filnamn 
    def filnamn_addera_suffix(self, filnamn, suffix, filtyp=".jpg"):
        filnamn = filnamn.lower()
        return(filnamn.replace(filtyp, suffix + filtyp))
    
    def filnamn_addera_prefix(self, filnamn, prefix):
        bara_filnamn = os.path.basename(filnamn)
        return(filnamn.replace(bara_filnamn, prefix + bara_filnamn))

    # skapa augmenterade bilder
    def augmentera_bild(self, ansikte_filnamn, bildmapp_ansikte, ansikte_beskuret):

        # skapa ImageDataGenerator objekt och definera augmenteringar och gränsvärden för dessa
        bild_generator = ImageDataGenerator(rotation_range = 60, horizontal_flip = True, brightness_range = (0.5, 1.5), zoom_range=[0.7,1])

        x = img_to_array(ansikte_beskuret)  
        x = x.reshape((1,) + x.shape)       
        
        # ta bort _A_ som indikerar orginalbild, samt filändelse eftersom flow lägger till filändelse
        ansikte_filnamn = ansikte_filnamn.replace("_A_", "_")
        ansikte_filnamn = ansikte_filnamn.lower()
        ansikte_filnamn = ansikte_filnamn.replace(".jpg", "")

        # skapa 3 slumpvist augmenterade bilder
        i = 0
        for batch in bild_generator.flow(x, batch_size=1, save_to_dir=bildmapp_ansikte, save_prefix=ansikte_filnamn, save_format='jpg'):
            i += 1
            if i > 2:
                break  

    # funktion för att om möjligt utöka MTCNN ansiktsbox med angiven andel, standard 20%, och beskära ut detta och returnerar bild på beskuret ansikte
    # detta då detta förbättrar träning/test noggranhet
    def beskär_ansikte(self, bild, bild_h, bild_b, y, h, x, b, utöka=0.2):

        h_utökning = round(h * 0.20)
        b_utökning = round(b * 0.20)
        if (y - (h_utökning // 2)) > 0 and (x - (b_utökning // 2) > 0) and (h + h_utökning < bild_h) and (b + b_utökning < bild_b):
            y = y - (h_utökning // 2)
            h = h + h_utökning
            x = x - (b_utökning // 2)
            b = b + b_utökning
        ansikte_beskuret = bild[y:y+h, x:x+b]

        return(ansikte_beskuret)

    # procedur som beskär flera ansikten från en bildfil mha MTCNN
    def beskär_ansikten(self, bildfil, bildmapp_ansikten, augmentera_bild=True, max_ansikten=5, min_pixels=2048):

        bild = cv2.imread(bildfil)
        bild_rgb = cv2.cvtColor(bild, cv2.COLOR_BGR2RGB)

        try:
            ansikten = self.detektor.detect_faces(bild_rgb)

        except Exception as e:
            print(f"Ett fel uppstod (bildfil: {bildfil}). Fel: {e}")   

        # sortera ansikten med högst confidence först, om vi har fler bilder än max gräns så ta de mest sannolika att det är ansikte först
        ansikten_sorterad = sorted(ansikten, reverse=True, key=lambda face: face['confidence'])
        
        i = 0
        antal_suddiga = 0
        antal_låg_upplösning = 0
        bild_h, bild_b, bild_kanaler = bild.shape

        # iterera igenom ansikten
        for ansikte in ansikten_sorterad:
            if i > max_ansikten:
                break
            x, y, b, h = ansikte['box']
            confidence = ansikte['confidence']
            upplösning = b * h
            if upplösning > min_pixels:
                i += 1
                är_suddig = False
                ansikte_beskuret = self.beskär_ansikte(bild, bild_h, bild_b, y, h, x, b, 0.2)
                ansikte_filnamn = self.filnamn_addera_suffix(bildfil, f"_A_{i}")  # lägg till _A_ för att kunna filtrera på orginal bilder (om vi ej vill använda augmenterade bilder) samt index för ansiktet
                # räkna ut varians av laplacian (skarpa kanter), tillsammans med confidence värdet kan vi flagga filer som suddiga
                suddighet = cv2.Laplacian(cv2.cvtColor(ansikte_beskuret, cv2.COLOR_RGB2BGR), cv2.CV_64F).var()
                if (suddighet < 50 and confidence < 0.99):
                    ansikte_filnamn = self.filnamn_addera_prefix(ansikte_filnamn, "__suddig_")
                    antal_suddiga += 1
                    är_suddig=True
                ansikte_filnamn = os.path.join(bildmapp_ansikten, os.path.basename(ansikte_filnamn))
                cv2.imwrite(ansikte_filnamn, ansikte_beskuret)

                # om augmenterade bilder önskas
                if augmentera_bild==True:
                    self.augmentera_bild(os.path.basename(ansikte_filnamn), bildmapp_ansikten, cv2.cvtColor(ansikte_beskuret, cv2.COLOR_RGB2BGR))

            else:
                antal_låg_upplösning += 1

            # uppdatera statistik
            self.bildbehandling_statistik_antal_ansiktsfiler_lågupplösning += antal_låg_upplösning
            self.bildbehandling_statistik_antal_ansiktsfiler_suddiga += antal_suddiga
            self.bildbehandling_statistik_antal_ansiktsfiler += i
            self.bildbehandling_statistik_antal_filer += 1

    # förbehandla bilder i mapp
    def bearbeta_mapp(self, bildmapp_källa, bildmapp_ansikten):
        
        self.nollställ_bildbehandling_statistik()

        # iteration av bildfiler i mapp
        for bild_fil in os.listdir(bildmapp_källa):
        
            thicks = 0
            try:
                bild_mapp_fil = os.path.join(bildmapp_källa, bild_fil)
                self.beskär_ansikten(bild_mapp_fil, bildmapp_ansikten)

                # anti cpu-hooging undvik att blockera datorn, kan justeras efter behov
                thicks += 1
                if thicks > 1:
                    time.sleep(1)
                    thicks = 0

            except Exception as e:
                print(f"Ett fel uppstod: {e}")   

        # skriv ut statistik för mappen
        print(f"Mapp: {os.path.basename(bildmapp_källa)} Antal filer: {self.bildbehandling_statistik_antal_filer} Antal ansiktsfiler: {self.bildbehandling_statistik_antal_ansiktsfiler} Antal suddiga: {self.bildbehandling_statistik_antal_ansiktsfiler_suddiga} Antal för låg upplösning: {self.bildbehandling_statistik_antal_ansiktsfiler_lågupplösning}")

class Noggranhet(keras.callbacks.Callback):

    def __init__(self, test_data, kategorier=[]):
        self.test_data = test_data
        self.kategorier = kategorier
        self.kategori_historik = []

    def on_epoch_end(self, epoch, logs=None):
        x_data, y_data = self.test_data

        # logga resultat efter varje epoch, räkna ut % rätt per kategori samt genomsnittlig tillförlitlighet på test
        korrekt = 0
        fel = 0

        resultat = self.model.predict(x_data, verbose=0)
        noggranhet_kategori = []

        for i in self.kategorier:
            self.kategori_historik.append([])

        kategori_korrekt = [0] * len(self.kategorier)
        kategori_fel = [0] * len(self.kategorier)

        for i in range(len(x_data)):
            x = x_data[i]
            y = y_data[i]

            res = resultat[i]
            kategori = np.argmax(y)
            förutsägelse_kategori = np.argmax(res)

            if(förutsägelse_kategori == kategori):  # test är korrekt
                noggranhet_kategori.append(["Korrekt:", str(y), str(res), str(förutsägelse_kategori), epoch])     
                kategori_korrekt[kategori] += 1   
                korrekt += 1
            else:       # test är fel
                noggranhet_kategori.append(["Fel:", str(y), str(res), str(förutsägelse_kategori), epoch])
                kategori_fel[kategori] += 1
                fel += 1

        print("")
        print("\tKorrekt: %d" %(korrekt))
        print("\tFel: %d" %(fel))

        for i in range(len(self.kategorier)):
            tot = float(kategori_korrekt[i] + kategori_fel[i])
            class_acc = -1
            if (tot > 0):
                class_acc = float(kategori_korrekt[i]) / tot

            print("\t%s: %.2f" %(self.kategorier[i],class_acc)) 

        acc = float(korrekt) / float(korrekt + fel)  

        print("\tTest tillförlitlighet: %.2f" %(acc))
