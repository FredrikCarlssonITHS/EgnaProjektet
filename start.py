import ai
import os

# ladda ner fil https://ithogskolan-my.sharepoint.com/:f:/g/personal/fredrik_carlsson_iths_se/Euy8q33H5lxPrFkOeo27iD4BTtxhpiwScYv0yQeTDp_14g?e=yzis4z
# och packa upp till en mapp på din dator. Ange sökväg nedan till resp mapp

mapp_huvud =  "C:/ITHS/Celebs" 
mapp_bilderförutspå = mapp_huvud + "/Analys"       #"C:/ITHS/Celebs/Analys"
mapp_orginalbilder = mapp_huvud + "/Orginal"       #"C:/ITHS/Celebs/Orginal"
mapp_ansiktsbilder = mapp_huvud + "/Ansikten"      #"C:/ITHS/Celebs/Ansikten"
mapp_modell = mapp_huvud + "/Modell"               #"C:/ITHS/Celebs/Modell"

modell = ai.Modell()

def meny():
    
    try:
        ai.rensa_skärm()
        print("=======================================================")
        print("Egna Projektet - Bildanalys v1.0 av Fredrik Carlsson")
        print("=======================================================")
        print("")
        print(f"Huvudmapp: {mapp_huvud}")
        print(f"Mapp bilder förutsägelse: {mapp_bilderförutspå}")
        print(f"Mapp bilder orginal: {mapp_orginalbilder}")
        print(f"Mapp bilder ansikten: {mapp_ansiktsbilder}")
        print(f"Mapp modell: {mapp_modell}")
        print("")
        print(f"Modell status: {modell}")
        print("")
        print("1) Förbehandla bilder")
        print("2) Träna modell")
        print("3) Ladda modell från fil")
        print("4) Förutspå en eller flera personer på en bild")
        print("A) Avsluta")
        alternativ = input(">")
        match (alternativ.upper()):
            case "1":
                FörbehandlaBilder()
            case "2":
                TränaModell()
            case "3":
                LaddaModell()
            case "4":
                FörutspåFleraPersoner()
            case "A":
                print("Tack för idag, välkommen åter !")
                exit()
            case _:
                print ("Felaktigt alternativ")
                input("Tryck ENTER för att återgå till meny.")
                meny()

    except Exception as e:
        print(f"Ett mycket allvarligt generellt fel uppstod: {e}")   


def FörbehandlaBilder():

    mappar = [i for i in os.listdir(mapp_orginalbilder) if os.path.isdir(os.path.join(mapp_orginalbilder, i))]

    print("Mappar att förbehandla bilder från => till")
    print()

    for mapp in mappar:
        print(f"Från {os.path.join(mapp_orginalbilder, mapp)} => {os.path.join(mapp_ansiktsbilder,mapp)}")

    alternativ = input("Fortsätt? J/N (ENTER=Ja) >")    
    if alternativ.upper() == "J" or alternativ == "":
        for mapp in mappar:
            modell.bearbeta_mapp(os.path.join(mapp_orginalbilder, mapp), os.path.join(mapp_ansiktsbilder,mapp))        
        print()
        input("Förbehandling av bilder klar. Tryck ENTER för att återgå till meny.")
    
    meny()

def TränaModell():

    alternativ = input("Ange antal epocher (ENTER=20) >")    
    if alternativ == "":
        antal_epoch = 20
    else:
        antal_epoch = int(alternativ)

    alternativ = input("Ange batch storlek (ENTER=32) >")    
    if alternativ == "":
        batch_storlek = 32
    else:
        batch_storlek = int(alternativ)

    alternativ = input("Ange antal filter (ENTER=32) >")    
    if alternativ == "":
        antal_filter = 32
    else:
        antal_filter = int(alternativ)        

    alternativ = input("Ange dropout (ENTER=0.5) >")    
    if alternativ == "":
        dropout = 0.5
    else:
        dropout = float(alternativ)        

    modell.ladda_data(mapp_ansiktsbilder, 0.8, 0.2, True, True)
    modell.normalisera()
    modell.skapa_modell(antal_filter, dropout)
    modell.träna(antal_epoch, batch_storlek)

    alternativ = input("Spara modell? J/N (ENTER=Ja) >")    
    if alternativ.upper() == "J" or alternativ == "":
        namn = input("Ange namn (ENTER=MinModell) >")    
        if namn == "":
            namn = "MinModell"
        modell.spara_modell(mapp_modell, namn)
        
    print()        
    input("Träning klar. Tryck ENTER för att återgå till meny.")
    meny()

def LaddaModell():

    filer = [f for f in os.listdir(mapp_modell) if os.path.isfile(os.path.join(mapp_modell, f)) and f.endswith(".keras") ]

    print("Ladda modell")
    print()
    print("Ange vilken modell du vill ladda")

    for fil in filer:
        print(f"{filer.index(fil)+1}. {fil}")

    print("A) Avbryt och återgå till meny")
    print()
    alternativ = input(">")
    if alternativ.upper() != "A":
        index = int(alternativ) - 1
        if index <= len(filer):
            fil = filer[index]
            fil = fil.replace(".keras", "")
            modell.ladda_modell(mapp_modell, fil)
            print(f"Modell {fil} laddad!")
        print()        
        input("Tryck ENTER för att återgå till meny.")
    meny()

def FörutspåFleraPersoner():

    if modell.modell_redo == True:

        filer = [f for f in os.listdir(mapp_bilderförutspå) if os.path.isfile(os.path.join(mapp_bilderförutspå, f)) and f.endswith(".jpg") ]

        print("Förutspå flera personer i en bild")
        print()
        print("Ange vilken bild du vill analysera")

        for fil in filer:
            print(f"{filer.index(fil)+1}. {fil}")

        print("A) Avbryt och återgå till meny")
        print()
        alternativ = input(">")
        if alternativ.upper() != "A":
            index = int(alternativ) - 1
            if index <= len(filer):
                fil = filer[index]
                modell.förutspå_multipla_ansikten(os.path.join(mapp_bilderförutspå, fil))
            print()        
            input("Tryck ENTER för att återgå till meny.")
        meny()

    else:
        print("Ingen modell redo. Kör träning eller ladda en sparad modell.")
        print()        
        input("Tryck ENTER för att återgå till meny.")
        meny()


meny()






