# Det egna projektet – Computer Vision för att hantera fotomappar
Av: Fredrik Carlsson

Problem

Tack vare mobilkameror och andra devices som GoPro, drönare mm samlas stora mängder foto osorterat som gör det svårt att hitta bilder. Det är svårt att söka eller hitta bilder på ett visst objekt eller person på ett effektivt sätt.



Lösningsförslag

Skapa en python app som med hjälp av en förtränad modell kan känna av generiska objekt, ex bil, trumpet, fotboll. Modellen ska kunna tränas för att känna igen personliga objekt, lokala fotbollslagets logotype, personer som familj, släkt och vänner. Dvs träna med hjälp av märkt data. Appen ska kunna bearbeta en eller flera fotomappar och tagga bilder dvs lägga till en EXIF tag som är sökbar i Windows utforskare så att man kan skriva sökord och få fram alla bilder som matchar i en miniatyr vy. 


Dataset

Egna bilder, för träning verkar det finnas en rekommendation om minst 200 bilder per objekt varav 80% för träning och 20% för test. I projektet kommer dock målet vara att se om det går att reducera denna mängden för att göra applikationen mer användarvänlig, ex genom en god förberedning av data (se nedan). En model som kan hitta ”main objekt” kan användas för att underlätta denna process och hitta lämpliga bilder för träning. Bilderna behöver därefter märkas, genom att flyttas till en undermapp döpt efter vad den innehåller. Mappens namn behöver encodas till ett numeriskt värde.
En del av appen kommer automatisera så mycket av data förberedelse inför träning. Modellen ska därefter kunna sparas och jämföras för att hitta optimala parametrar.


Förberedning av data

För att nå bästa resultat bör bilder normaliseras/standardiseras och augmenteras/varieras. 
-	Resize till en given storlek
-	Ansikte markeras eller crop:as (bilden beskärs så bara objektet, ex ansikte visas)
-	Augmentering genom att ändra vinkel på objekt, använda varierande bilder på samma objekt, för en person kan detta vara med och utan cykelhjälm, solglasögon och andra accessoarer.
-	För denna typ av bilder bör RGB färgkanaler behållas, för ex att korrekt identifiera fotbollslagets logotype/matchtröjor, ögonfärg, hårfärg.


Projekt steg

1.	Skapa python app som kan förbereda data, ex göra resize, hitta objekt, augmentera
2.	Ta fram eget dataset från lämpliga fotomappar, göra en grov sortering utifrån miniatyr vy i Utforskaren. Undvika att använda gamla inaktuella bilder, ex bild på en person som har ändrat stil, barn som växer etc.
3.	Köra python app för att förbereda data mot utvalda bilder
4.	Skapa python app som kan tränas på bilderna
5.	Träna och utvärdera med olika algorimter, layers och parametrar (epochs, batch size, dropouts mfl)
6.	Sätt ihop slutgiltig python app som kan appliceras på bildmappar och tagga bilder. Appen ska även kunna presentera en rapport över en körning och vilken precision. Problematiska bilder ska presenteras för åtgärd, ex utökad träning.
7.	Dokumentera
