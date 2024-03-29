# Noms : Guillaume Comte et Jonathan Lo
# Data Mining
# Final Project
# Juin 2019

import sys
import btk
import numpy as np
import glob
from sklearn import tree
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

path = "/home/guillaume/Bureau/PROJET DATA MINING OPERATIONAL/Gait/Sofamehack2019/Sub_DB_Checked/CP/*"
#path = "/home/jonathanlo/Documents/DataMining/DM_Final_Project/Sofamehack2019/Sofamehack2019/Sub_DB_Checked/CP/*"
#pathSave = '/home/jonathanlo/Documents/DataMining/DM_Final_Project/Resultats/'
pathSave = '/home/guillaume/Bureau/PROJET DATA MINING OPERATIONAL/Resultats/'

filesCP = glob.glob(path) # Pour avoir tous les noms de fichiers dans une liste
allerror = []
X = []
Y = []
third = int(round(len(filesCP)/3)) # Deux tiers du nombre de fichiers

training_from = 0
training_to = third
training_from2 = third
training_to2 = third*2
testing_from = third*2
testing_to = len(filesCP)

leventFr = []
llabel = []
lcontext = []

llnew = []
llnec = []
llnel = []

allrcp = []
allrfd = []
allritw = []




def dataPreparation_CP():
    for f in range(training_from,training_to):
        reader = btk.btkAcquisitionFileReader()
        reader.SetFilename(filesCP[f])
        reader.Update()
        acq = reader.GetOutput() # acq is the btk aquisition object
        ptFr = acq.GetPointFrequency() # give the point frequency
        nbFrame = acq.GetPointFrameNumber() # give the number of frames
        metadata = acq.GetMetaData()
        nbEvent = metadata.FindChild("EVENT").value().FindChild("USED").value().GetInfo().ToInt()[0]
        # Contient tous les points de chaque capteurs
        LeftHeel = acq.GetPoint("LHEE").GetValues()
        RightHeel = acq.GetPoint("RHEE").GetValues()
        LeftAnkle = acq.GetPoint("LANK").GetValues()
        RightAnkle = acq.GetPoint("RANK").GetValues()
        LeftToe = acq.GetPoint("LTOE").GetValues()
        RightToe = acq.GetPoint("RTOE").GetValues()
        LeftTib = acq.GetPoint("LTIB").GetValues()
        RightTib = acq.GetPoint("RTIB").GetValues()
        for i in range(len(LeftHeel)): # Nous n'utilisons pas la profondeur de la marche
            LeftHeel[i][1] = 10
            LeftAnkle[i][1] = 10
            LeftToe[i][1] = 10
            LeftTib[i][1] = 10
            RightHeel[i][1] = 40
            RightAnkle[i][1] = 40
            RightToe[i][1] = 40
            RightTib[i][1] = 40

        for i in range(0,nbEvent):
            event = acq.GetEvent(i) # extract the first event of the aquisition
            label = event.GetLabel() # return a string representing the Label
            context = event.GetContext() # return a string representing the Context
            eventFr = event.GetFrame() # return the frame as an integer
            # Chercher les frames :
            LHeel = LeftHeel[eventFr-1,:]
            RHeel = RightHeel[eventFr-1,:]
            LAnkle = LeftAnkle[eventFr-1,:]
            RAnkle = RightAnkle[eventFr-1,:]
            LToe = LeftToe[eventFr-1,:]
            RToe = RightToe[eventFr-1,:]
            LTib = LeftTib[eventFr-1,:]
            RTib = RightTib[eventFr-1,:]
            allValues = [] # Stocke les frames des evenements
            allValues.extend(LHeel)
            allValues.extend(RHeel)
            allValues.extend(LAnkle)
            allValues.extend(RAnkle)
            allValues.extend(LToe)
            allValues.extend(RToe)
            allValues.extend(LTib)
            allValues.extend(RTib)
            X.append(allValues) # Les frames des evenements
            Y.append(label + " " + context) # Les labels des evenements (les labels correspondants)



    for f in range(training_from2,training_to2):
        reader = btk.btkAcquisitionFileReader()
        reader.SetFilename(filesCP[f])
        reader.Update()
        acq = reader.GetOutput() # acq is the btk aquisition object
        ptFr = acq.GetPointFrequency() # give the point frequency
        nbFrame = acq.GetPointFrameNumber() # give the number of frames
        metadata = acq.GetMetaData()
        nbEvent = metadata.FindChild("EVENT").value().FindChild("USED").value().GetInfo().ToInt()[0]
        # Contient tous les points de chaque capteurs
        LeftHeel = acq.GetPoint("LHEE").GetValues()
        RightHeel = acq.GetPoint("RHEE").GetValues()
        LeftAnkle = acq.GetPoint("LANK").GetValues()
        RightAnkle = acq.GetPoint("RANK").GetValues()
        LeftToe = acq.GetPoint("LTOE").GetValues()
        RightToe = acq.GetPoint("RTOE").GetValues()
        LeftTib = acq.GetPoint("LTIB").GetValues()
        RightTib = acq.GetPoint("RTIB").GetValues()
        for i in range(len(LeftHeel)): # Nous n'utilisons pas la profondeur de la marche
            LeftHeel[i][1] = 10
            LeftAnkle[i][1] = 10
            LeftToe[i][1] = 10
            LeftTib[i][1] = 10
            RightHeel[i][1] = 40
            RightAnkle[i][1] = 40
            RightToe[i][1] = 40
            RightTib[i][1] = 40

        for i in range(0,nbEvent):
            event = acq.GetEvent(i) # extract the first event of the aquisition
            label = event.GetLabel() # return a string representing the Label
            context = event.GetContext() # return a string representing the Context
            eventFr = event.GetFrame() # return the frame as an integer
            # Chercher les frames :
            LHeel = LeftHeel[eventFr-1,:]
            RHeel = RightHeel[eventFr-1,:]
            LAnkle = LeftAnkle[eventFr-1,:]
            RAnkle = RightAnkle[eventFr-1,:]
            LToe = LeftToe[eventFr-1,:]
            RToe = RightToe[eventFr-1,:]
            LTib = LeftTib[eventFr-1,:]
            RTib = RightTib[eventFr-1,:]
            allValues = [] # Stocke les frames des evenements
            allValues.extend(LHeel)
            allValues.extend(RHeel)
            allValues.extend(LAnkle)
            allValues.extend(RAnkle)
            allValues.extend(LToe)
            allValues.extend(RToe)
            allValues.extend(LTib)
            allValues.extend(RTib)
            X.append(allValues) # Les frames des evenements
            Y.append(label + " " + context)



    # Gestion des nothing
    for i in range(0,nbEvent):
        event = acq.GetEvent(i) # extract the first event of the aquisition
        label = event.GetLabel() # return a string representing the Label
        context = event.GetContext() # return a string representing the Context
        eventFr = event.GetFrame() # return the frame as an integer
        for val in [-10,-9,-8,-7,-6,-5,-4,-3,-2,2,3,4,5,6,7,8,9,10]:
            if eventFr + val < nbFrame and eventFr + val > 1:
                LHeel = LeftHeel[eventFr + val,:]
                RHeel = RightHeel[eventFr + val,:]
                LAnkle = LeftAnkle[eventFr + val,:]
                RAnkle = RightAnkle[eventFr + val,:]
                LToe = LeftToe[eventFr + val,:]
                RToe = RightToe[eventFr + val,:]
                LTib = LeftTib[eventFr + val,:]
                RTib = RightTib[eventFr + val,:]
                allValues = []
                allValues.extend(LHeel)
                allValues.extend(RHeel)
                allValues.extend(LAnkle)
                allValues.extend(RAnkle)
                allValues.extend(LToe)
                allValues.extend(RToe)
                allValues.extend(LTib)
                allValues.extend(RTib)
                X.append(allValues)
                Y.append("nothing") # PROBLEM ICI






def prediction_CP():
    #clf = tree.DecisionTreeClassifier()
    #clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5,2),random_state=1)
    #clf = GaussianNB()
    clf = KNeighborsClassifier(2)
    #clf = svm.SVC(gamma=0.001, C=100.)

    clf = clf.fit(X, Y)


    # Testing
    for f in range(testing_from,testing_to):
        indexname = filesCP[f].split("/",9)
        fname = indexname[len(indexname)-1]
        reader = btk.btkAcquisitionFileReader() # acq is the btk aquisition object
        reader.SetFilename(filesCP[f])
        reader.Update()
        acq = reader.GetOutput()
        nbFrame = acq.GetPointFrameNumber()

        LeftHeel = acq.GetPoint("LHEE").GetValues()
        RightHeel = acq.GetPoint("RHEE").GetValues()
        LeftAnkle = acq.GetPoint("LANK").GetValues()
        RightAnkle = acq.GetPoint("RANK").GetValues()
        LeftToe = acq.GetPoint("LTOE").GetValues()
        RightToe = acq.GetPoint("RTOE").GetValues()
        LeftTib = acq.GetPoint("LTIB").GetValues()
        RightTib = acq.GetPoint("RTIB").GetValues()
        for i in range(len(LeftHeel)):
            LeftHeel[i][1] = 10
            LeftAnkle[i][1] = 10
            LeftToe[i][1] = 10
            LeftTib[i][1] = 10
            RightHeel[i][1] = 40
            RightAnkle[i][1] = 40
            RightToe[i][1] = 40
            RightTib[i][1] = 40

        prediction = [[]]*(nbFrame-1)

        for i in range(1,nbFrame):
            LHeel = LeftHeel[i-1,:]
            RHeel = RightHeel[i-1,:]
            LAnkle = LeftAnkle[i-1,:]
            RAnkle = RightAnkle[i-1,:]
            LToe = LeftToe[i-1,:]
            RToe = RightToe[i-1,:]
            LTib = LeftTib[i-1,:]
            RTib = RightTib[i-1,:]
            allValues = []
            allValues.extend(LHeel)
            allValues.extend(RHeel)
            allValues.extend(LAnkle)
            allValues.extend(RAnkle)
            allValues.extend(LToe)
            allValues.extend(RToe)
            allValues.extend(LTib)
            allValues.extend(RTib)
            prediction[i-1] = allValues


        newValues = clf.predict(prediction)

        ev = 0
        lerror = []
        lnewevent = []
        lnec = []
        lnel = []

        for i in range(0,nbFrame-2):
            if (newValues[i] != "nothing"):
                f = i
                j = 1
                tmpl = newValues[i].split(" ",1)
                if i + 50 > nbFrame-1:
                    value = (nbFrame - i -1)
                else:
                    value = 50
                for k in range(i,i+value):
                    if newValues[k] != "nothing":
                        if tmpl[1] == newValues[k].split(" ",1)[1]:
                            newValues[k] = "nothing"
                min = 10000000
                max = -1000000
                if tmpl[0] == "Foot_Strike_GS":
                    for k in range(i,i+value):
                        if LeftHeel[k][2] + RightHeel[k][2] < min:
                            min = LeftToe[k][2] + RightToe[k][2]
                            f = k
                if tmpl[0] == "Foot_Off_GS":
                    for k in range(i,i+value):
                        if LeftHeel[k][2] + RightHeel[k][2] < max:
                            max = LeftHeel[k][2] + RightHeel[k][2]
                            f = k
                lnec.append(tmpl[1])
                lnel.append(tmpl[0])
                event=btk.btkEvent()
                event.SetLabel(tmpl[0] )
                event.SetContext(tmpl[1])
                event.SetId(1)
                event.SetFrame(np.round(f))
                event.SetTime(f/100.0)
                acq.AppendEvent(event)
                ev = ev +1
                lnewevent.append(f)


        llnec.append(lnec)
        llnel.append(lnel)
        llnew.append(lnewevent)

        writer = btk.btkAcquisitionFileWriter()
        writer.SetInput(acq)
        writer.SetFilename(pathSave + fname)
        writer.Update()

        reader = btk.btkAcquisitionFileReader()
        reader.SetFilename(pathSave + fname)
        reader.Update()
        acq = reader.GetOutput() # acq is the btk aquisition object
        ptFr = acq.GetPointFrequency() # give the point frequency
        nbFrame = acq.GetPointFrameNumber() # give the number of frames
        metadata = acq.GetMetaData()
        nbEvent = metadata.FindChild("EVENT").value().FindChild("USED").value().GetInfo().ToInt()[0]

        for i in range(0,nbEvent-1):
            event = acq.GetEvent(i) # extract the first event of the aquisition
            label = event.GetLabel() # return a string representing the Label
            context = event.GetContext() # return a string representing the Context
            eventFr = event.GetFrame() # return the frame as an integer
            if context == "Left" and acq.GetEvent(i+1).GetContext() == "Left":
                if (label == "Foot_Strike_GS" and acq.GetEvent(i+1).GetLabel() == "Foot_Strike_GS"):
                    f = np.round((acq.GetEvent(i+1).GetFrame() - eventFr)/2)
                    lnec.append("Left")
                    lnel.append("Foot_Off_GS")
                    event=btk.btkEvent()
                    event.SetLabel("Foot_Off_GS")
                    event.SetContext("Left")
                    event.SetId(1)
                    event.SetFrame(np.round(f))
                    event.SetTime(f/100.0)
                    acq.AppendEvent(event)
                    lnewevent.append(f)
                if (label == "Foot_Off_GS" and acq.GetEvent(i+1).GetLabel() == "Foot_Off_GS"):
                    f = np.round((acq.GetEvent(i+1).GetFrame() - eventFr)/2)
                    lnec.append("Left")
                    lnel.append("Foot_Strike_GS")
                    event=btk.btkEvent()
                    event.SetLabel("Foot_Strike_GS")
                    event.SetContext("Left")
                    event.SetId(1)
                    event.SetFrame(np.round(f))
                    event.SetTime(f/100.0)
                    acq.AppendEvent(event)
                    lnewevent.append(f)
            if context == "Right" and acq.GetEvent(i+1).GetContext() == "Right":
                if (label == "Foot_Strike_GS" and acq.GetEvent(i+1).GetLabel() == "Foot_Strike_GS"):
                    f = np.round((acq.GetEvent(i+1).GetFrame() - eventFr)/2)
                    lnec.append("Right")
                    lnel.append("Foot_Off_GS")
                    event=btk.btkEvent()
                    event.SetLabel("Foot_Off_GS")
                    event.SetContext("Right")
                    event.SetId(1)
                    event.SetFrame(np.round(f))
                    event.SetTime(f/100.0)
                    acq.AppendEvent(event)
                    lnewevent.append(f)
                if (label == "Foot_Off_GS" and acq.GetEvent(i+1).GetLabel() == "Foot_Off_GS"):
                    f = np.round((acq.GetEvent(i+1).GetFrame() - eventFr)/2)
                    lnec.append("Right")
                    lnel.append("Foot_Strike_GS")
                    event=btk.btkEvent()
                    event.SetLabel("Foot_Strike_GS")
                    event.SetContext("Right")
                    event.SetId(1)
                    event.SetFrame(np.round(f))
                    event.SetTime(f/100.0)
                    acq.AppendEvent(event)
                    lnewevent.append(f)
            if context == "Left" and acq.GetEvent(i+1).GetContext() == "Left":
                if (label == "Foot_Strike_GS" and acq.GetEvent(i+1).GetLabel() == "Foot_Strike_GS"):
                    f = np.round((acq.GetEvent(i+1).GetFrame() - eventFr)/2)
                    lnec.append("Left")
                    lnel.append("Foot_Off_GS")
                    event=btk.btkEvent()
                    event.SetLabel("Foot_Off_GS")
                    event.SetContext("Left")
                    event.SetId(1)
                    event.SetFrame(np.round(f))
                    event.SetTime(f/100.0)
                    acq.AppendEvent(event)
                    lnewevent.append(f)
                if (label == "Foot_Off_GS" and acq.GetEvent(i+1).GetLabel() == "Foot_Off_GS"):
                    f = np.round((acq.GetEvent(i+1).GetFrame() - eventFr)/2)
                    lnec.append("Left")
                    lnel.append("Foot_Strike_GS")
                    event=btk.btkEvent()
                    event.SetLabel("Foot_Strike_GS")
                    event.SetContext("Left")
                    event.SetId(1)
                    event.SetFrame(np.round(f))
                    event.SetTime(f/100.0)
                    acq.AppendEvent(event)
                    lnewevent.append(f)
            if context == "Right" and acq.GetEvent(i+1).GetContext() == "Left":
                if (label == "Foot_Off_GS" and acq.GetEvent(i+1).GetLabel() == "Foot_Strke_GS"):
                    f = np.round((acq.GetEvent(i+1).GetFrame() - eventFr)/2)
                    lnec.append("Right")
                    lnel.append("Foot_Off_GS")
                    event=btk.btkEvent()
                    event.SetLabel("Foot_Off_GS")
                    event.SetContext("Right")
                    event.SetId(1)
                    event.SetFrame(np.round(f))
                    event.SetTime(f/100.0)
                    acq.AppendEvent(event)
                    lnewevent.append(f)
            if context == "Left" and acq.GetEvent(i+1).GetContext() == "Right":
                if (label == "Foot_Off_GS" and acq.GetEvent(i+1).GetLabel() == "Foot_Strike_GS"):
                    f = np.round((acq.GetEvent(i+1).GetFrame() - eventFr)/2)
                    lnec.append("Left")
                    lnel.append("Foot_Off_GS")
                    event=btk.btkEvent()
                    event.SetLabel("Foot_Off_GS")
                    event.SetContext("Right")
                    event.SetId(1)
                    event.SetFrame(np.round(f))
                    event.SetTime(f/100.0)
                    acq.AppendEvent(event)
                    lnewevent.append(f)

        leventFr = []
        llabel = []
        lcontext = []
        for i in range(0,nbEvent):
            event = acq.GetEvent(i)
            eventFr = event.GetFrame() # return the frame as an integer
            leventFr.append(eventFr)
            label = event.GetLabel() # return a string representing the Label
            context = event.GetContext() # return a string representing the Context
            llabel.append(label)
            lcontext.append(context)

        for o in range(0,len(leventFr)):
            mindist = 45
            for p in range(0,len(lnewevent)):
                if lnel[p] == llabel[o] and lnec[p] == lcontext[o]:
                    if abs(leventFr[o] - lnewevent[p]) < mindist :
                        mindist = abs(leventFr[o] - lnewevent[p])
            lerror.append(mindist)
        if ev > 100:
            print("ALERT")

        writer = btk.btkAcquisitionFileWriter()
        writer.SetInput(acq)
        writer.SetFilename(pathSave + fname)

        writer.Update()
        allerror.append(lerror)
    valerr = 0
    for k in allerror:
        for p in k:
            if p > 1:
                valerr = valerr + np.exp(p)

    print(" ")
    print("error CP :")
    print(valerr/len(allerror))
    allrcp.append(valerr/len(allerror))




def dataPreparation_ITW():
    for f in range(training_from,training_to):
        reader = btk.btkAcquisitionFileReader()
        reader.SetFilename(filesCP[f])
        reader.Update()
        acq = reader.GetOutput() # acq is the btk aquisition object
        ptFr = acq.GetPointFrequency() # give the point frequency
        nbFrame = acq.GetPointFrameNumber() # give the number of frames
        metadata = acq.GetMetaData()
        nbEvent = metadata.FindChild("EVENT").value().FindChild("USED").value().GetInfo().ToInt()[0]
        # Contient tous les points de chaque capteurs
        LeftHeel = acq.GetPoint("LHEE").GetValues()
        RightHeel = acq.GetPoint("RHEE").GetValues()
        LeftAnkle = acq.GetPoint("LANK").GetValues()
        RightAnkle = acq.GetPoint("RANK").GetValues()
        LeftToe = acq.GetPoint("LTOE").GetValues()
        RightToe = acq.GetPoint("RTOE").GetValues()
        LeftTib = acq.GetPoint("LTIB").GetValues()
        RightTib = acq.GetPoint("RTIB").GetValues()
        for i in range(len(LeftHeel)): # Nous n'utilisons pas la profondeur de la marche
            LeftHeel[i][1] = 10
            LeftAnkle[i][1] = 10
            LeftToe[i][1] = 10
            LeftTib[i][1] = 10
            RightHeel[i][1] = 40
            RightAnkle[i][1] = 40
            RightToe[i][1] = 40
            RightTib[i][1] = 40

        for i in range(0,nbEvent):
            event = acq.GetEvent(i) # extract the first event of the aquisition
            label = event.GetLabel() # return a string representing the Label
            context = event.GetContext() # return a string representing the Context
            eventFr = event.GetFrame() # return the frame as an integer
            # Chercher les frames :
            LHeel = LeftHeel[eventFr-1,:]
            RHeel = RightHeel[eventFr-1,:]
            LAnkle = LeftAnkle[eventFr-1,:]
            RAnkle = RightAnkle[eventFr-1,:]
            LToe = LeftToe[eventFr-1,:]
            RToe = RightToe[eventFr-1,:]
            LTib = LeftTib[eventFr-1,:]
            RTib = RightTib[eventFr-1,:]
            allValues = [] # Stocke les frames des evenements
            allValues.extend(LHeel)
            allValues.extend(RHeel)
            allValues.extend(LAnkle)
            allValues.extend(RAnkle)
            allValues.extend(LToe)
            allValues.extend(RToe)
            allValues.extend(LTib)
            allValues.extend(RTib)
            X.append(allValues) # Les frames des evenements
            Y.append(label + " " + context) # Les labels des evenements (les labels correspondants)



for f in range(training_from2,training_to2):
    reader = btk.btkAcquisitionFileReader()
    reader.SetFilename(filesCP[f])
    reader.Update()
    acq = reader.GetOutput() # acq is the btk aquisition object
    ptFr = acq.GetPointFrequency() # give the point frequency
    nbFrame = acq.GetPointFrameNumber() # give the number of frames
    metadata = acq.GetMetaData()
    nbEvent = metadata.FindChild("EVENT").value().FindChild("USED").value().GetInfo().ToInt()[0]
    # Contient tous les points de chaque capteurs
    LeftHeel = acq.GetPoint("LHEE").GetValues()
    RightHeel = acq.GetPoint("RHEE").GetValues()
    LeftAnkle = acq.GetPoint("LANK").GetValues()
    RightAnkle = acq.GetPoint("RANK").GetValues()
    LeftToe = acq.GetPoint("LTOE").GetValues()
    RightToe = acq.GetPoint("RTOE").GetValues()
    LeftTib = acq.GetPoint("LTIB").GetValues()
    RightTib = acq.GetPoint("RTIB").GetValues()
    for i in range(len(LeftHeel)): # Nous n'utilisons pas la profondeur de la marche
        LeftHeel[i][1] = 10
        LeftAnkle[i][1] = 10
        LeftToe[i][1] = 10
        LeftTib[i][1] = 10
        RightHeel[i][1] = 40
        RightAnkle[i][1] = 40
        RightToe[i][1] = 40
        RightTib[i][1] = 40

    for i in range(0,nbEvent):
        event = acq.GetEvent(i) # extract the first event of the aquisition
        label = event.GetLabel() # return a string representing the Label
        context = event.GetContext() # return a string representing the Context
        eventFr = event.GetFrame() # return the frame as an integer
        # Chercher les frames :
        LHeel = LeftHeel[eventFr-1,:]
        RHeel = RightHeel[eventFr-1,:]
        LAnkle = LeftAnkle[eventFr-1,:]
        RAnkle = RightAnkle[eventFr-1,:]
        LToe = LeftToe[eventFr-1,:]
        RToe = RightToe[eventFr-1,:]
        LTib = LeftTib[eventFr-1,:]
        RTib = RightTib[eventFr-1,:]
        allValues = [] # Stocke les frames des evenements
        allValues.extend(LHeel)
        allValues.extend(RHeel)
        allValues.extend(LAnkle)
        allValues.extend(RAnkle)
        allValues.extend(LToe)
        allValues.extend(RToe)
        allValues.extend(LTib)
        allValues.extend(RTib)
        X.append(allValues) # Les frames des evenements
        Y.append(label + " " + context)

    # Gestion des nothing
    for i in range(0,nbEvent):
        event = acq.GetEvent(i) # extract the first event of the aquisition
        label = event.GetLabel() # return a string representing the Label
        context = event.GetContext() # return a string representing the Context
        eventFr = event.GetFrame() # return the frame as an integer
        for val in [-10,-9,-8,-7,-6,-5,-4,-3,-2,2,3,4,5,6,7,8,9,10]:
            if eventFr + val < nbFrame and eventFr + val > 1:
                LHeel = LeftHeel[eventFr + val,:]
                RHeel = RightHeel[eventFr + val,:]
                LAnkle = LeftAnkle[eventFr + val,:]
                RAnkle = RightAnkle[eventFr + val,:]
                LToe = LeftToe[eventFr + val,:]
                RToe = RightToe[eventFr + val,:]
                LTib = LeftTib[eventFr + val,:]
                RTib = RightTib[eventFr + val,:]
                allValues = []
                allValues.extend(LHeel)
                allValues.extend(RHeel)
                allValues.extend(LAnkle)
                allValues.extend(RAnkle)
                allValues.extend(LToe)
                allValues.extend(RToe)
                allValues.extend(LTib)
                allValues.extend(RTib)
                X.append(allValues)
                Y.append("nothing") # PROBLEM ICI

def prediction_ITW():
    #clf = tree.DecisionTreeClassifier()
    #clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5,2),random_state=1)
    #clf = GaussianNB()
    clf = KNeighborsClassifier(2)
    #clf = svm.SVC(gamma=0.001, C=100.)

    clf = clf.fit(X, Y)
    # Testing
    for f in range(testing_from,testing_to):
        indexname = filesCP[f].split("/",9)
        fname = indexname[len(indexname)-1]
        reader = btk.btkAcquisitionFileReader() # acq is the btk aquisition object
        reader.SetFilename(filesCP[f])
        reader.Update()
        acq = reader.GetOutput()
        nbFrame = acq.GetPointFrameNumber()

        LeftHeel = acq.GetPoint("LHEE").GetValues()
        RightHeel = acq.GetPoint("RHEE").GetValues()
        LeftAnkle = acq.GetPoint("LANK").GetValues()
        RightAnkle = acq.GetPoint("RANK").GetValues()
        LeftToe = acq.GetPoint("LTOE").GetValues()
        RightToe = acq.GetPoint("RTOE").GetValues()
        LeftTib = acq.GetPoint("LTIB").GetValues()
        RightTib = acq.GetPoint("RTIB").GetValues()
        for i in range(len(LeftHeel)):
            LeftHeel[i][1] = 10
            LeftAnkle[i][1] = 10
            LeftToe[i][1] = 10
            LeftTib[i][1] = 10
            RightHeel[i][1] = 40
            RightAnkle[i][1] = 40
            RightToe[i][1] = 40
            RightTib[i][1] = 40

        prediction = [[]]*(nbFrame-1)

        for i in range(1,nbFrame):
            LHeel = LeftHeel[i-1,:]
            RHeel = RightHeel[i-1,:]
            LAnkle = LeftAnkle[i-1,:]
            RAnkle = RightAnkle[i-1,:]
            LToe = LeftToe[i-1,:]
            RToe = RightToe[i-1,:]
            LTib = LeftTib[i-1,:]
            RTib = RightTib[i-1,:]
            allValues = []
            allValues.extend(LHeel)
            allValues.extend(RHeel)
            allValues.extend(LAnkle)
            allValues.extend(RAnkle)
            allValues.extend(LToe)
            allValues.extend(RToe)
            allValues.extend(LTib)
            allValues.extend(RTib)
            prediction[i-1] = allValues

        newValues = clf.predict(prediction)

        ev = 0
        lerror = []
        lnewevent = []
        lnec = []
        lnel = []

        for i in range(0,nbFrame-2):
            if (newValues[i] != "nothing"):
                f = i
                j = 1
                tmpl = newValues[i].split(" ",1)
                if i + 50 > nbFrame-1:
                    value = (nbFrame - i -1)
                else:
                    value = 50
                for k in range(i,i+value):
                    if newValues[k] != "nothing":
                        if tmpl[1] == newValues[k].split(" ",1)[1]:
                            newValues[k] = "nothing"
                min = 10000000
                max = -1000000
                if tmpl[0] == "Foot_Strike_GS":
                    for k in range(i,i+value):
                        # if LeftToe[k][2] + RightToe[k][2] + LeftHeel[k][2] + RightHeel[k][2] < min:
                        if LeftHeel[k][2] + RightHeel[k][2] < min:
                            min = LeftToe[k][2] + RightToe[k][2]
                            f = k
                if tmpl[0] == "Foot_Off_GS":
                    for k in range(i,i+value):
                        if LeftHeel[k][2] + RightHeel[k][2] < max:
                            max = LeftHeel[k][2] + RightHeel[k][2]
                            f = k
                lnec.append(tmpl[1])
                lnel.append(tmpl[0])
                event=btk.btkEvent()
                event.SetLabel(tmpl[0] )
                event.SetContext(tmpl[1])
                event.SetId(1)
                event.SetFrame(np.round(f))
                event.SetTime(f/100.0)
                acq.AppendEvent(event)
                ev = ev +1
                lnewevent.append(f)

        llnec.append(lnec)
        llnel.append(lnel)
        llnew.append(lnewevent)

        writer = btk.btkAcquisitionFileWriter()
        writer.SetInput(acq)
        writer.SetFilename(pathSave + fname)
        writer.Update()

        reader = btk.btkAcquisitionFileReader()
        reader.SetFilename(pathSave + fname)
        reader.Update()
        acq = reader.GetOutput() # acq is the btk aquisition object
        ptFr = acq.GetPointFrequency() # give the point frequency
        nbFrame = acq.GetPointFrameNumber() # give the number of frames
        metadata = acq.GetMetaData()
        nbEvent = metadata.FindChild("EVENT").value().FindChild("USED").value().GetInfo().ToInt()[0]

        for i in range(0,nbEvent-1):
            event = acq.GetEvent(i) # extract the first event of the aquisition
            label = event.GetLabel() # return a string representing the Label
            context = event.GetContext() # return a string representing the Context
            eventFr = event.GetFrame() # return the frame as an integer
            if context == "Left" and acq.GetEvent(i+1).GetContext() == "Left":
                if (label == "Foot_Strike_GS" and acq.GetEvent(i+1).GetLabel() == "Foot_Strike_GS"):
                    f = np.round((acq.GetEvent(i+1).GetFrame() - eventFr)/2)
                    lnec.append("Left")
                    lnel.append("Foot_Off_GS")
                    event=btk.btkEvent()
                    event.SetLabel("Foot_Off_GS")
                    event.SetContext("Left")
                    event.SetId(1)
                    event.SetFrame(np.round(f))
                    event.SetTime(f/100.0)
                    acq.AppendEvent(event)
                    lnewevent.append(f)
                if (label == "Foot_Off_GS" and acq.GetEvent(i+1).GetLabel() == "Foot_Off_GS"):
                    f = np.round((acq.GetEvent(i+1).GetFrame() - eventFr)/2)
                    lnec.append("Left")
                    lnel.append("Foot_Strike_GS")
                    event=btk.btkEvent()
                    event.SetLabel("Foot_Strike_GS")
                    event.SetContext("Left")
                    event.SetId(1)
                    event.SetFrame(np.round(f))
                    event.SetTime(f/100.0)
                    acq.AppendEvent(event)
                    lnewevent.append(f)
            if context == "Right" and acq.GetEvent(i+1).GetContext() == "Right":
                if (label == "Foot_Strike_GS" and acq.GetEvent(i+1).GetLabel() == "Foot_Strike_GS"):
                    f = np.round((acq.GetEvent(i+1).GetFrame() - eventFr)/2)
                    lnec.append("Right")
                    lnel.append("Foot_Off_GS")
                    event=btk.btkEvent()
                    event.SetLabel("Foot_Off_GS")
                    event.SetContext("Right")
                    event.SetId(1)
                    event.SetFrame(np.round(f))
                    event.SetTime(f/100.0)
                    acq.AppendEvent(event)
                    lnewevent.append(f)
                if (label == "Foot_Off_GS" and acq.GetEvent(i+1).GetLabel() == "Foot_Off_GS"):
                    f = np.round((acq.GetEvent(i+1).GetFrame() - eventFr)/2)
                    lnec.append("Right")
                    lnel.append("Foot_Strike_GS")
                    event=btk.btkEvent()
                    event.SetLabel("Foot_Strike_GS")
                    event.SetContext("Right")
                    event.SetId(1)
                    event.SetFrame(np.round(f))
                    event.SetTime(f/100.0)
                    acq.AppendEvent(event)
                    lnewevent.append(f)
            if context == "Left" and acq.GetEvent(i+1).GetContext() == "Left":
                if (label == "Foot_Strike_GS" and acq.GetEvent(i+1).GetLabel() == "Foot_Strike_GS"):
                    f = np.round((acq.GetEvent(i+1).GetFrame() - eventFr)/2)
                    lnec.append("Left")
                    lnel.append("Foot_Off_GS")
                    event=btk.btkEvent()
                    event.SetLabel("Foot_Off_GS")
                    event.SetContext("Left")
                    event.SetId(1)
                    event.SetFrame(np.round(f))
                    event.SetTime(f/100.0)
                    acq.AppendEvent(event)
                    lnewevent.append(f)
                if (label == "Foot_Off_GS" and acq.GetEvent(i+1).GetLabel() == "Foot_Off_GS"):
                    f = np.round((acq.GetEvent(i+1).GetFrame() - eventFr)/2)
                    lnec.append("Left")
                    lnel.append("Foot_Strike_GS")
                    event=btk.btkEvent()
                    event.SetLabel("Foot_Strike_GS")
                    event.SetContext("Left")
                    event.SetId(1)
                    event.SetFrame(np.round(f))
                    event.SetTime(f/100.0)
                    acq.AppendEvent(event)
                    lnewevent.append(f)
            if context == "Right" and acq.GetEvent(i+1).GetContext() == "Left":
                if (label == "Foot_Off_GS" and acq.GetEvent(i+1).GetLabel() == "Foot_Strke_GS"):
                    f = np.round((acq.GetEvent(i+1).GetFrame() - eventFr)/2)
                    lnec.append("Right")
                    lnel.append("Foot_Off_GS")
                    event=btk.btkEvent()
                    event.SetLabel("Foot_Off_GS")
                    event.SetContext("Right")
                    event.SetId(1)
                    event.SetFrame(np.round(f))
                    event.SetTime(f/100.0)
                    acq.AppendEvent(event)
                    lnewevent.append(f)
            if context == "Left" and acq.GetEvent(i+1).GetContext() == "Right":
                if (label == "Foot_Off_GS" and acq.GetEvent(i+1).GetLabel() == "Foot_Strike_GS"):
                    f = np.round((acq.GetEvent(i+1).GetFrame() - eventFr)/2)
                    lnec.append("Left")
                    lnel.append("Foot_Off_GS")
                    event=btk.btkEvent()
                    event.SetLabel("Foot_Off_GS")
                    event.SetContext("Right")
                    event.SetId(1)
                    event.SetFrame(np.round(f))
                    event.SetTime(f/100.0)
                    acq.AppendEvent(event)
                    lnewevent.append(f)

        leventFr = []
        llabel = []
        lcontext = []
        for i in range(0,nbEvent):
            event = acq.GetEvent(i)
            eventFr = event.GetFrame() # return the frame as an integer
            leventFr.append(eventFr)
            label = event.GetLabel() # return a string representing the Label
            context = event.GetContext() # return a string representing the Context
            llabel.append(label)
            lcontext.append(context)

        for o in range(0,len(leventFr)):
            mindist = 45
            for p in range(0,len(lnewevent)):
                if lnel[p] == llabel[o] and lnec[p] == lcontext[o]:
                    if abs(leventFr[o] - lnewevent[p]) < mindist :
                        mindist = abs(leventFr[o] - lnewevent[p])
            lerror.append(mindist)
        if ev > 100:
            print("ALERT")

        writer = btk.btkAcquisitionFileWriter()
        writer.SetInput(acq)
        writer.SetFilename(pathSave + fname)

        writer.Update()
        allerror.append(lerror)
    valerr = 0
    for k in allerror:
        for p in k:
            if p > 1:
                valerr = valerr + np.exp(p)

    print(" ")
    print("error ITW :")
    print(valerr/len(allerror))
    allritw.append(valerr/len(allerror))




def dataPreparation_FD():
    for f in range(training_from,training_to):
        reader = btk.btkAcquisitionFileReader()
        reader.SetFilename(filesCP[f])
        reader.Update()
        acq = reader.GetOutput() # acq is the btk aquisition object
        ptFr = acq.GetPointFrequency() # give the point frequency
        nbFrame = acq.GetPointFrameNumber() # give the number of frames
        metadata = acq.GetMetaData()
        nbEvent = metadata.FindChild("EVENT").value().FindChild("USED").value().GetInfo().ToInt()[0]
        # Contient tous les points de chaque capteurs
        LeftHeel = acq.GetPoint("LHEE").GetValues()
        RightHeel = acq.GetPoint("RHEE").GetValues()
        LeftAnkle = acq.GetPoint("LANK").GetValues()
        RightAnkle = acq.GetPoint("RANK").GetValues()
        LeftToe = acq.GetPoint("LTOE").GetValues()
        RightToe = acq.GetPoint("RTOE").GetValues()
        LeftTib = acq.GetPoint("LTIB").GetValues()
        RightTib = acq.GetPoint("RTIB").GetValues()
        for i in range(len(LeftHeel)): # Nous n'utilisons pas la profondeur de la marche
            LeftHeel[i][1] = 10
            LeftAnkle[i][1] = 10
            LeftToe[i][1] = 10
            LeftTib[i][1] = 10
            RightHeel[i][1] = 40
            RightAnkle[i][1] = 40
            RightToe[i][1] = 40
            RightTib[i][1] = 40

        for i in range(0,nbEvent):
            event = acq.GetEvent(i) # extract the first event of the aquisition
            label = event.GetLabel() # return a string representing the Label
            context = event.GetContext() # return a string representing the Context
            eventFr = event.GetFrame() # return the frame as an integer
            # Chercher les frames :
            LHeel = LeftHeel[eventFr-1,:]
            RHeel = RightHeel[eventFr-1,:]
            LAnkle = LeftAnkle[eventFr-1,:]
            RAnkle = RightAnkle[eventFr-1,:]
            LToe = LeftToe[eventFr-1,:]
            RToe = RightToe[eventFr-1,:]
            LTib = LeftTib[eventFr-1,:]
            RTib = RightTib[eventFr-1,:]
            allValues = [] # Stocke les frames des evenements
            allValues.extend(LHeel)
            allValues.extend(RHeel)
            allValues.extend(LAnkle)
            allValues.extend(RAnkle)
            allValues.extend(LToe)
            allValues.extend(RToe)
            allValues.extend(LTib)
            allValues.extend(RTib)
            X.append(allValues) # Les frames des evenements
            Y.append(label + " " + context) # Les labels des evenements (les labels correspondants)

for f in range(training_from2,training_to2):
    reader = btk.btkAcquisitionFileReader()
    reader.SetFilename(filesCP[f])
    reader.Update()
    acq = reader.GetOutput() # acq is the btk aquisition object
    ptFr = acq.GetPointFrequency() # give the point frequency
    nbFrame = acq.GetPointFrameNumber() # give the number of frames
    metadata = acq.GetMetaData()
    nbEvent = metadata.FindChild("EVENT").value().FindChild("USED").value().GetInfo().ToInt()[0]
    # Contient tous les points de chaque capteurs
    LeftHeel = acq.GetPoint("LHEE").GetValues()
    RightHeel = acq.GetPoint("RHEE").GetValues()
    LeftAnkle = acq.GetPoint("LANK").GetValues()
    RightAnkle = acq.GetPoint("RANK").GetValues()
    LeftToe = acq.GetPoint("LTOE").GetValues()
    RightToe = acq.GetPoint("RTOE").GetValues()
    LeftTib = acq.GetPoint("LTIB").GetValues()
    RightTib = acq.GetPoint("RTIB").GetValues()
    for i in range(len(LeftHeel)): # Nous n'utilisons pas la profondeur de la marche
        LeftHeel[i][1] = 10
        LeftAnkle[i][1] = 10
        LeftToe[i][1] = 10
        LeftTib[i][1] = 10
        RightHeel[i][1] = 40
        RightAnkle[i][1] = 40
        RightToe[i][1] = 40
        RightTib[i][1] = 40

    for i in range(0,nbEvent):
        event = acq.GetEvent(i) # extract the first event of the aquisition
        label = event.GetLabel() # return a string representing the Label
        context = event.GetContext() # return a string representing the Context
        eventFr = event.GetFrame() # return the frame as an integer
        # Chercher les frames :
        LHeel = LeftHeel[eventFr-1,:]
        RHeel = RightHeel[eventFr-1,:]
        LAnkle = LeftAnkle[eventFr-1,:]
        RAnkle = RightAnkle[eventFr-1,:]
        LToe = LeftToe[eventFr-1,:]
        RToe = RightToe[eventFr-1,:]
        LTib = LeftTib[eventFr-1,:]
        RTib = RightTib[eventFr-1,:]
        allValues = [] # Stocke les frames des evenements
        allValues.extend(LHeel)
        allValues.extend(RHeel)
        allValues.extend(LAnkle)
        allValues.extend(RAnkle)
        allValues.extend(LToe)
        allValues.extend(RToe)
        allValues.extend(LTib)
        allValues.extend(RTib)
        X.append(allValues) # Les frames des evenements
        Y.append(label + " " + context)


    # Gestion des nothing
    for i in range(0,nbEvent):
        event = acq.GetEvent(i) # extract the first event of the aquisition
        label = event.GetLabel() # return a string representing the Label
        context = event.GetContext() # return a string representing the Context
        eventFr = event.GetFrame() # return the frame as an integer
        for val in [-10,-9,-8,-7,-6,-5,-4,-3,-2,2,3,4,5,6,7,8,9,10]:
            if eventFr + val < nbFrame and eventFr + val > 1:
                LHeel = LeftHeel[eventFr + val,:]
                RHeel = RightHeel[eventFr + val,:]
                LAnkle = LeftAnkle[eventFr + val,:]
                RAnkle = RightAnkle[eventFr + val,:]
                LToe = LeftToe[eventFr + val,:]
                RToe = RightToe[eventFr + val,:]
                LTib = LeftTib[eventFr + val,:]
                RTib = RightTib[eventFr + val,:]
                allValues = []
                allValues.extend(LHeel)
                allValues.extend(RHeel)
                allValues.extend(LAnkle)
                allValues.extend(RAnkle)
                allValues.extend(LToe)
                allValues.extend(RToe)
                allValues.extend(LTib)
                allValues.extend(RTib)
                X.append(allValues)
                Y.append("nothing") # PROBLEM ICI

def prediction_FD():
    #clf = tree.DecisionTreeClassifier()
    #clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5,2),random_state=1)
    #clf = GaussianNB()
    clf = KNeighborsClassifier(2)
    #clf = svm.SVC(gamma=0.001, C=100.)

    clf = clf.fit(X, Y)


    # Testing
    for f in range(testing_from,testing_to):
        indexname = filesCP[f].split("/",9)
        fname = indexname[len(indexname)-1]
        reader = btk.btkAcquisitionFileReader() # acq is the btk aquisition object
        reader.SetFilename(filesCP[f])
        reader.Update()
        acq = reader.GetOutput()
        nbFrame = acq.GetPointFrameNumber()

        LeftHeel = acq.GetPoint("LHEE").GetValues()
        RightHeel = acq.GetPoint("RHEE").GetValues()
        LeftAnkle = acq.GetPoint("LANK").GetValues()
        RightAnkle = acq.GetPoint("RANK").GetValues()
        LeftToe = acq.GetPoint("LTOE").GetValues()
        RightToe = acq.GetPoint("RTOE").GetValues()
        LeftTib = acq.GetPoint("LTIB").GetValues()
        RightTib = acq.GetPoint("RTIB").GetValues()
        for i in range(len(LeftHeel)):
            LeftHeel[i][1] = 10
            LeftAnkle[i][1] = 10
            LeftToe[i][1] = 10
            LeftTib[i][1] = 10
            RightHeel[i][1] = 40
            RightAnkle[i][1] = 40
            RightToe[i][1] = 40
            RightTib[i][1] = 40

        prediction = [[]]*(nbFrame-1)

        for i in range(1,nbFrame):
            LHeel = LeftHeel[i-1,:]
            RHeel = RightHeel[i-1,:]
            LAnkle = LeftAnkle[i-1,:]
            RAnkle = RightAnkle[i-1,:]
            LToe = LeftToe[i-1,:]
            RToe = RightToe[i-1,:]
            LTib = LeftTib[i-1,:]
            RTib = RightTib[i-1,:]
            allValues = []
            allValues.extend(LHeel)
            allValues.extend(RHeel)
            allValues.extend(LAnkle)
            allValues.extend(RAnkle)
            allValues.extend(LToe)
            allValues.extend(RToe)
            allValues.extend(LTib)
            allValues.extend(RTib)
            prediction[i-1] = allValues

        newValues = clf.predict(prediction)

        ev = 0
        lerror = []
        lnewevent = []
        lnec = []
        lnel = []

        for i in range(0,nbFrame-2):
            if (newValues[i] != "nothing"):
                f = i
                j = 1
                tmpl = newValues[i].split(" ",1)
                if i + 50 > nbFrame-1:
                    value = (nbFrame - i -1)
                else:
                    value = 50
                for k in range(i,i+value):
                    if newValues[k] != "nothing":
                        if tmpl[1] == newValues[k].split(" ",1)[1]:
                            newValues[k] = "nothing"
                min = 10000000
                max = -1000000
                if tmpl[0] == "Foot_Strike_GS":
                    for k in range(i,i+value):
                        if LeftHeel[k][2] + RightHeel[k][2] < min:
                            min = LeftToe[k][2] + RightToe[k][2]
                            f = k
                if tmpl[0] == "Foot_Off_GS":
                    for k in range(i,i+value):
                        if LeftHeel[k][2] + RightHeel[k][2] < max:
                            max = LeftHeel[k][2] + RightHeel[k][2]
                            f = k
                lnec.append(tmpl[1])
                lnel.append(tmpl[0])
                event=btk.btkEvent()
                event.SetLabel(tmpl[0] )
                event.SetContext(tmpl[1])
                event.SetId(1)
                event.SetFrame(np.round(f))
                event.SetTime(f/100.0)
                acq.AppendEvent(event)
                ev = ev +1
                lnewevent.append(f)

        llnec.append(lnec)
        llnel.append(lnel)
        llnew.append(lnewevent)

        writer = btk.btkAcquisitionFileWriter()
        writer.SetInput(acq)
        writer.SetFilename(pathSave + fname)
        writer.Update()

        reader = btk.btkAcquisitionFileReader()
        reader.SetFilename(pathSave + fname)
        reader.Update()
        acq = reader.GetOutput() # acq is the btk aquisition object
        ptFr = acq.GetPointFrequency() # give the point frequency
        nbFrame = acq.GetPointFrameNumber() # give the number of frames
        metadata = acq.GetMetaData()
        nbEvent = metadata.FindChild("EVENT").value().FindChild("USED").value().GetInfo().ToInt()[0]

        for i in range(0,nbEvent-1):
            event = acq.GetEvent(i) # extract the first event of the aquisition
            label = event.GetLabel() # return a string representing the Label
            context = event.GetContext() # return a string representing the Context
            eventFr = event.GetFrame() # return the frame as an integer
            if context == "Left" and acq.GetEvent(i+1).GetContext() == "Left":
                if (label == "Foot_Strike_GS" and acq.GetEvent(i+1).GetLabel() == "Foot_Strike_GS"):
                    f = np.round((acq.GetEvent(i+1).GetFrame() - eventFr)/2)
                    lnec.append("Left")
                    lnel.append("Foot_Off_GS")
                    event=btk.btkEvent()
                    event.SetLabel("Foot_Off_GS")
                    event.SetContext("Left")
                    event.SetId(1)
                    event.SetFrame(np.round(f))
                    event.SetTime(f/100.0)
                    acq.AppendEvent(event)
                    lnewevent.append(f)
                if (label == "Foot_Off_GS" and acq.GetEvent(i+1).GetLabel() == "Foot_Off_GS"):
                    f = np.round((acq.GetEvent(i+1).GetFrame() - eventFr)/2)
                    lnec.append("Left")
                    lnel.append("Foot_Strike_GS")
                    event=btk.btkEvent()
                    event.SetLabel("Foot_Strike_GS")
                    event.SetContext("Left")
                    event.SetId(1)
                    event.SetFrame(np.round(f))
                    event.SetTime(f/100.0)
                    acq.AppendEvent(event)
                    lnewevent.append(f)
            if context == "Right" and acq.GetEvent(i+1).GetContext() == "Right":
                if (label == "Foot_Strike_GS" and acq.GetEvent(i+1).GetLabel() == "Foot_Strike_GS"):
                    f = np.round((acq.GetEvent(i+1).GetFrame() - eventFr)/2)
                    lnec.append("Right")
                    lnel.append("Foot_Off_GS")
                    event=btk.btkEvent()
                    event.SetLabel("Foot_Off_GS")
                    event.SetContext("Right")
                    event.SetId(1)
                    event.SetFrame(np.round(f))
                    event.SetTime(f/100.0)
                    acq.AppendEvent(event)
                    lnewevent.append(f)
                if (label == "Foot_Off_GS" and acq.GetEvent(i+1).GetLabel() == "Foot_Off_GS"):
                    f = np.round((acq.GetEvent(i+1).GetFrame() - eventFr)/2)
                    lnec.append("Right")
                    lnel.append("Foot_Strike_GS")
                    event=btk.btkEvent()
                    event.SetLabel("Foot_Strike_GS")
                    event.SetContext("Right")
                    event.SetId(1)
                    event.SetFrame(np.round(f))
                    event.SetTime(f/100.0)
                    acq.AppendEvent(event)
                    lnewevent.append(f)
            if context == "Left" and acq.GetEvent(i+1).GetContext() == "Left":
                if (label == "Foot_Strike_GS" and acq.GetEvent(i+1).GetLabel() == "Foot_Strike_GS"):
                    f = np.round((acq.GetEvent(i+1).GetFrame() - eventFr)/2)
                    lnec.append("Left")
                    lnel.append("Foot_Off_GS")
                    event=btk.btkEvent()
                    event.SetLabel("Foot_Off_GS")
                    event.SetContext("Left")
                    event.SetId(1)
                    event.SetFrame(np.round(f))
                    event.SetTime(f/100.0)
                    acq.AppendEvent(event)
                    lnewevent.append(f)
                if (label == "Foot_Off_GS" and acq.GetEvent(i+1).GetLabel() == "Foot_Off_GS"):
                    f = np.round((acq.GetEvent(i+1).GetFrame() - eventFr)/2)
                    lnec.append("Left")
                    lnel.append("Foot_Strike_GS")
                    event=btk.btkEvent()
                    event.SetLabel("Foot_Strike_GS")
                    event.SetContext("Left")
                    event.SetId(1)
                    event.SetFrame(np.round(f))
                    event.SetTime(f/100.0)
                    acq.AppendEvent(event)
                    lnewevent.append(f)
            if context == "Right" and acq.GetEvent(i+1).GetContext() == "Left":
                if (label == "Foot_Off_GS" and acq.GetEvent(i+1).GetLabel() == "Foot_Strke_GS"):
                    f = np.round((acq.GetEvent(i+1).GetFrame() - eventFr)/2)
                    lnec.append("Right")
                    lnel.append("Foot_Off_GS")
                    event=btk.btkEvent()
                    event.SetLabel("Foot_Off_GS")
                    event.SetContext("Right")
                    event.SetId(1)
                    event.SetFrame(np.round(f))
                    event.SetTime(f/100.0)
                    acq.AppendEvent(event)
                    lnewevent.append(f)
            if context == "Left" and acq.GetEvent(i+1).GetContext() == "Right":
                if (label == "Foot_Off_GS" and acq.GetEvent(i+1).GetLabel() == "Foot_Strike_GS"):
                    f = np.round((acq.GetEvent(i+1).GetFrame() - eventFr)/2)
                    lnec.append("Left")
                    lnel.append("Foot_Off_GS")
                    event=btk.btkEvent()
                    event.SetLabel("Foot_Off_GS")
                    event.SetContext("Right")
                    event.SetId(1)
                    event.SetFrame(np.round(f))
                    event.SetTime(f/100.0)
                    acq.AppendEvent(event)
                    lnewevent.append(f)

        leventFr = []
        llabel = []
        lcontext = []
        for i in range(0,nbEvent):
            event = acq.GetEvent(i)
            eventFr = event.GetFrame() # return the frame as an integer
            leventFr.append(eventFr)
            label = event.GetLabel() # return a string representing the Label
            context = event.GetContext() # return a string representing the Context
            llabel.append(label)
            lcontext.append(context)

        for o in range(0,len(leventFr)):
            mindist = 45
            for p in range(0,len(lnewevent)):
                if lnel[p] == llabel[o] and lnec[p] == lcontext[o]:
                    if abs(leventFr[o] - lnewevent[p]) < mindist :
                        mindist = abs(leventFr[o] - lnewevent[p])
            lerror.append(mindist)
        if ev > 100:
            print("ALERT")

        writer = btk.btkAcquisitionFileWriter()
        writer.SetInput(acq)
        #writer.SetFilename('/home/guillaume/Bureau/PROJET DATA MINING OPERATIONAL/Results/CP/' + fname)
        writer.SetFilename(pathSave + fname)
        writer.Update()
        allerror.append(lerror)
    valerr = 0
    for k in allerror:
        for p in k:
            if p > 1:
                valerr = valerr + np.exp(p)

    print(" ")
    print("error FD :")
    print(valerr/len(allerror))
    allrfd.append(valerr/len(allerror))



###############MAIN
print("########## CP ############")
print("Premier tiers")
dataPreparation_CP()
prediction_CP()
 # second trial
print("Deuxieme tiers")
filesCP = glob.glob(path)
X = []
Y = []
training_from = third
training_to = third*2
training_from2 = third*2
training_to2 = len(filesCP)
testing_from = 0
testing_to = third
dataPreparation_CP()
prediction_CP()
#third trial
print("Troisieme tiers")
filesCP = glob.glob(path)
X = []
Y = []
training_from = 0
training_to = third
training_from2 = third*2
training_to2 = len(filesCP)
testing_from = third
testing_to = third*2
dataPreparation_CP()
prediction_CP()



print("########## ITW ############")
print("Premier tiers")
path = "/home/guillaume/Bureau/PROJET DATA MINING OPERATIONAL/Gait/Sofamehack2019/Sub_DB_Checked/ITW/*"
filesCP = glob.glob(path)
X = []
Y = []
third = int(round(len(filesCP)/3))
training_from = 0
training_to = third
training_from2 = third
training_to2 = third*2
testing_from = third*2
testing_to = len(filesCP)
dataPreparation_ITW()
prediction_ITW()
 # second trial
print("Second tiers")
X = []
Y = []
training_from = third
training_to = third*2
training_from2 = third*2
training_to2 = len(filesCP)
testing_from = 0
testing_to = third
dataPreparation_ITW()
prediction_ITW()
#third trial
print("Troisieme tiers")
X = []
Y = []
training_from = 0
training_to = third
training_from2 = third*2
training_to2 = len(filesCP)
testing_from = third
testing_to = third*2
dataPreparation_ITW()
prediction_ITW()



print("########## FD ############")
print("Premier tiers")
path = "/home/guillaume/Bureau/PROJET DATA MINING OPERATIONAL/Gait/Sofamehack2019/Sub_DB_Checked/FD/*"
filesCP = glob.glob(path)
X = []
Y = []
third = int(round(len(filesCP)/3))
training_from = 0
training_to = third
training_from2 = third
training_to2 = third*2
testing_from = third*2
testing_to = len(filesCP)
dataPreparation_FD()
prediction_FD()
 # second trial
print("Second tiers")
filesCP = glob.glob(path)
X = []
Y = []
training_from = third
training_to = third*2
training_from2 = third*2
training_to2 = len(filesCP)
testing_from = 0
testing_to = third
dataPreparation_FD()
prediction_FD()
#third trial
print("Troisieme tiers")
filesCP = glob.glob(path)
X = []
Y = []
training_from = 0
training_to = third
training_from2 = third*2
training_to2 = len(filesCP)
testing_from = third
testing_to = third*2
dataPreparation_FD()
prediction_FD()

print(" ")
print(" ")
print("Global Error CP :")
print(np.mean(allrcp))
print(" ")
print("Global Error ITW :")
print(np.mean(allritw))
print(" ")
print("Global Error FD :")
print(np.mean(allrfd))
