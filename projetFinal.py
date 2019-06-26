# Noms : Jonathan Lo et Guillaume Comte
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

#pathCP = "/home/guillaume/Bureau/PROJET DATA MINING OPERATIONAL/Gait/Sofamehack2019/Sub_DB_Checked/CP/*"
path = "/home/jonathanlo/Documents/DataMining/DM_Final_Project/Sofamehack2019/Sofamehack2019/Sub_DB_Checked/CP/*"
pathSave = '/home/jonathanlo/Documents/DataMining/DM_Final_Project/Resultats/'
#pathSave = '/home/guillaume/Bureau/PROJET DATA MINING OPERATIONAL/Results/CP/'

filesCP = glob.glob(path) # Pour avoir tous les noms de fichiers dans une liste
allerror = []
X = []
Y = []
third = int(round(len(filesCP)/3))*2 # Deux tiers du nombre de fichiers
#third = len(filesCP)-1

leventFr = []
llabel = []
lcontext = []



llnew = []
llnec = []
llnel = []





def dataPreparation_CP():
    # path = "/home/jonathanlo/Documents/DataMining/DM_Final_Project/Sofamehack2019/Sofamehack2019/Sub_DB_Checked/"+pathology+"/*"
    # filesCP = glob.glob(path) # Pour avoir tous les noms de fichiers dans une liste
    # X = []
    # Y = []
    # third = int(round(len(filesCP)/3))*2 # Deux tiers du nombre de fichiers

    for f in range(0,third):
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
        # LeftKnee = acq.GetPoint("LKNE").GetValues()
        # RightKnee = acq.GetPoint("RKNE").GetValues()
        # LeftThigh = acq.GetPoint("LTHI").GetValues()
        # RightThigh = acq.GetPoint("RTHI").GetValues()
        LeftTib = acq.GetPoint("LTIB").GetValues()
        RightTib = acq.GetPoint("RTIB").GetValues()
        for i in range(len(LeftHeel)): # Nous n'utilisons pas la profondeur de la marche
            LeftHeel[i][1] = 10
            LeftAnkle[i][1] = 10
            LeftToe[i][1] = 10
            # LeftKnee[i][1] = 10
            # LeftThigh[i][1] = 10
            LeftTib[i][1] = 10
            RightHeel[i][1] = 40
            RightAnkle[i][1] = 40
            RightToe[i][1] = 40
            # RightKnee[i][1] = 40
            # RightThigh[i][1] = 40
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
            # LKnee = LeftKnee[eventFr-1,:]
            # RKnee = RightKnee[eventFr-1,:]
            # LThigh = LeftThigh[eventFr-1,:]
            # RThigh = RightThigh[eventFr-1,:]
            LTib = LeftTib[eventFr-1,:]
            RTib = RightTib[eventFr-1,:]
            allValues = [] # Stocke les frames des evenements
            allValues.extend(LHeel)
            allValues.extend(RHeel)
            allValues.extend(LAnkle)
            allValues.extend(RAnkle)
            allValues.extend(LToe)
            allValues.extend(RToe)
            # allValues.extend(LKnee)
            # allValues.extend(RKnee)
            # allValues.extend(LThigh)
            # allValues.extend(RThigh)
            allValues.extend(LTib)
            allValues.extend(RTib)
            X.append(allValues) # Les frames des evenements
            Y.append(label + " " + context) # Les labels des evenements (les labels correspondants)



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
                # LKnee = LeftKnee[event + val,:]
                # RKnee = RightKnee[event + val,:]
                # LThigh = LeftThigh[event + val,:]
                # RThigh = RightThigh[event + val,:]
                LTib = LeftTib[eventFr + val,:]
                RTib = RightTib[eventFr + val,:]
                allValues = []
                allValues.extend(LHeel)
                allValues.extend(RHeel)
                allValues.extend(LAnkle)
                allValues.extend(RAnkle)
                allValues.extend(LToe)
                allValues.extend(RToe)
                # allValues.extend(LKnee)
                # allValues.extend(RKnee)
                # allValues.extend(LThigh)
                # allValues.extend(RThigh)
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
    for f in range(third,len(filesCP)):
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
        # LeftKnee = acq.GetPoint("LKNE").GetValues()
        # RightKnee = acq.GetPoint("RKNE").GetValues()
        # LeftThigh = acq.GetPoint("LTHI").GetValues()
        # RightThigh = acq.GetPoint("RTHI").GetValues()
        LeftTib = acq.GetPoint("LTIB").GetValues()
        RightTib = acq.GetPoint("RTIB").GetValues()
        for i in range(len(LeftHeel)):
            LeftHeel[i][1] = 10
            LeftAnkle[i][1] = 10
            LeftToe[i][1] = 10
            # LeftKnee[i][1] = 10
            # LeftThigh[i][1] = 10
            LeftTib[i][1] = 10
            RightHeel[i][1] = 40
            RightAnkle[i][1] = 40
            RightToe[i][1] = 40
            # RightKnee[i][1] = 40
            # RightThigh[i][1] = 40
            RightTib[i][1] = 40

        prediction = [[]]*(nbFrame-1)

        for i in range(1,nbFrame):
            LHeel = LeftHeel[i-1,:]
            RHeel = RightHeel[i-1,:]
            LAnkle = LeftAnkle[i-1,:]
            RAnkle = RightAnkle[i-1,:]
            LToe = LeftToe[i-1,:]
            RToe = RightToe[i-1,:]
            # LKnee = LeftKnee[i-1,:]
            # RKnee = RightKnee[i-1,:]
            # LThigh = LeftThigh[i-1,:]
            # RThigh = RightThigh[i-1,:]
            LTib = LeftTib[i-1,:]
            RTib = RightTib[i-1,:]
            allValues = []
            allValues.extend(LHeel)
            allValues.extend(RHeel)
            allValues.extend(LAnkle)
            allValues.extend(RAnkle)
            allValues.extend(LToe)
            allValues.extend(RToe)
            # allValues.extend(LKnee)
            # allValues.extend(RKnee)
            # allValues.extend(LThigh)
            # allValues.extend(RThigh)
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
        #writer.SetFilename('/home/guillaume/Bureau/PROJET DATA MINING OPERATIONAL/Results/CP/' + fname)
        writer.SetFilename(pathSave + fname)
        writer.Update()

        reader = btk.btkAcquisitionFileReader()
        #reader.SetFilename('/home/guillaume/Bureau/PROJET DATA MINING OPERATIONAL/Results/CP/' + fname)
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



        # print(leventFr)
        # print(llabel)
        # print(lcontext)


        # for i in range(0,len[levent[d]]):
        #     event = acq.GetEvent(i)
        #     eventFr = event.GetFrame() # return the frame as an integer
        #     leventFr.append(eventFr)
        #     label = event.GetLabel() # return a string representing the Label
        #     context = event.GetContext() # return a string representing the Context
        #     llabel.append(label)
        #     lcontext.append(context)
        # for o in range(0,len(leventFr)):
        #     mindist = nbFrame
        #     for p in range(0,len(lnewevent)):
        #         if lnel[p] == llabel[o] and lnec[p] == lcontext[o]:
        #             if abs(leventFr[o] - lnewevent[p]) < mindist :
        #                 mindist = abs(leventFr[o] - lnewevent[p])
        #     lerror.append(mindist)
        # if ev > 100:
        #     print("ALERT")


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
    # print(allerror)
    valerr = 0
    for k in allerror:
        for p in k:
            if p > 1:
                valerr = valerr + np.exp(p)

    print(" ")
    print("error CP :")
    print(valerr/len(allerror))




def dataPreparation_ITW():
    # path = "/home/jonathanlo/Documents/DataMining/DM_Final_Project/Sofamehack2019/Sofamehack2019/Sub_DB_Checked/"+pathology+"/*"
    # filesCP = glob.glob(path) # Pour avoir tous les noms de fichiers dans une liste
    # X = []
    # Y = []
    # third = int(round(len(filesCP)/3))*2 # Deux tiers du nombre de fichiers

    for f in range(0,third):
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
        # LeftKnee = acq.GetPoint("LKNE").GetValues()
        # RightKnee = acq.GetPoint("RKNE").GetValues()
        # LeftThigh = acq.GetPoint("LTHI").GetValues()
        # RightThigh = acq.GetPoint("RTHI").GetValues()
        LeftTib = acq.GetPoint("LTIB").GetValues()
        RightTib = acq.GetPoint("RTIB").GetValues()
        for i in range(len(LeftHeel)): # Nous n'utilisons pas la profondeur de la marche
            LeftHeel[i][1] = 10
            LeftAnkle[i][1] = 10
            LeftToe[i][1] = 10
            # LeftKnee[i][1] = 10
            # LeftThigh[i][1] = 10
            LeftTib[i][1] = 10
            RightHeel[i][1] = 40
            RightAnkle[i][1] = 40
            RightToe[i][1] = 40
            # RightKnee[i][1] = 40
            # RightThigh[i][1] = 40
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
            # LKnee = LeftKnee[eventFr-1,:]
            # RKnee = RightKnee[eventFr-1,:]
            # LThigh = LeftThigh[eventFr-1,:]
            # RThigh = RightThigh[eventFr-1,:]
            LTib = LeftTib[eventFr-1,:]
            RTib = RightTib[eventFr-1,:]
            allValues = [] # Stocke les frames des evenements
            allValues.extend(LHeel)
            allValues.extend(RHeel)
            allValues.extend(LAnkle)
            allValues.extend(RAnkle)
            allValues.extend(LToe)
            allValues.extend(RToe)
            # allValues.extend(LKnee)
            # allValues.extend(RKnee)
            # allValues.extend(LThigh)
            # allValues.extend(RThigh)
            allValues.extend(LTib)
            allValues.extend(RTib)
            X.append(allValues) # Les frames des evenements
            Y.append(label + " " + context) # Les labels des evenements (les labels correspondants)



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
                # LKnee = LeftKnee[event + val,:]
                # RKnee = RightKnee[event + val,:]
                # LThigh = LeftThigh[event + val,:]
                # RThigh = RightThigh[event + val,:]
                LTib = LeftTib[eventFr + val,:]
                RTib = RightTib[eventFr + val,:]
                allValues = []
                allValues.extend(LHeel)
                allValues.extend(RHeel)
                allValues.extend(LAnkle)
                allValues.extend(RAnkle)
                allValues.extend(LToe)
                allValues.extend(RToe)
                # allValues.extend(LKnee)
                # allValues.extend(RKnee)
                # allValues.extend(LThigh)
                # allValues.extend(RThigh)
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
    for f in range(third,len(filesCP)):
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
        # LeftKnee = acq.GetPoint("LKNE").GetValues()
        # RightKnee = acq.GetPoint("RKNE").GetValues()
        # LeftThigh = acq.GetPoint("LTHI").GetValues()
        # RightThigh = acq.GetPoint("RTHI").GetValues()
        LeftTib = acq.GetPoint("LTIB").GetValues()
        RightTib = acq.GetPoint("RTIB").GetValues()
        for i in range(len(LeftHeel)):
            LeftHeel[i][1] = 10
            LeftAnkle[i][1] = 10
            LeftToe[i][1] = 10
            # LeftKnee[i][1] = 10
            # LeftThigh[i][1] = 10
            LeftTib[i][1] = 10
            RightHeel[i][1] = 40
            RightAnkle[i][1] = 40
            RightToe[i][1] = 40
            # RightKnee[i][1] = 40
            # RightThigh[i][1] = 40
            RightTib[i][1] = 40

        prediction = [[]]*(nbFrame-1)

        for i in range(1,nbFrame):
            LHeel = LeftHeel[i-1,:]
            RHeel = RightHeel[i-1,:]
            LAnkle = LeftAnkle[i-1,:]
            RAnkle = RightAnkle[i-1,:]
            LToe = LeftToe[i-1,:]
            RToe = RightToe[i-1,:]
            # LKnee = LeftKnee[i-1,:]
            # RKnee = RightKnee[i-1,:]
            # LThigh = LeftThigh[i-1,:]
            # RThigh = RightThigh[i-1,:]
            LTib = LeftTib[i-1,:]
            RTib = RightTib[i-1,:]
            allValues = []
            allValues.extend(LHeel)
            allValues.extend(RHeel)
            allValues.extend(LAnkle)
            allValues.extend(RAnkle)
            allValues.extend(LToe)
            allValues.extend(RToe)
            # allValues.extend(LKnee)
            # allValues.extend(RKnee)
            # allValues.extend(LThigh)
            # allValues.extend(RThigh)
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
        #writer.SetFilename('/home/guillaume/Bureau/PROJET DATA MINING OPERATIONAL/Results/CP/' + fname)
        writer.SetFilename(pathSave + fname)
        writer.Update()

        reader = btk.btkAcquisitionFileReader()
        #reader.SetFilename('/home/guillaume/Bureau/PROJET DATA MINING OPERATIONAL/Results/CP/' + fname)
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



        # print(leventFr)
        # print(llabel)
        # print(lcontext)


        # for i in range(0,len[levent[d]]):
        #     event = acq.GetEvent(i)
        #     eventFr = event.GetFrame() # return the frame as an integer
        #     leventFr.append(eventFr)
        #     label = event.GetLabel() # return a string representing the Label
        #     context = event.GetContext() # return a string representing the Context
        #     llabel.append(label)
        #     lcontext.append(context)
        # for o in range(0,len(leventFr)):
        #     mindist = nbFrame
        #     for p in range(0,len(lnewevent)):
        #         if lnel[p] == llabel[o] and lnec[p] == lcontext[o]:
        #             if abs(leventFr[o] - lnewevent[p]) < mindist :
        #                 mindist = abs(leventFr[o] - lnewevent[p])
        #     lerror.append(mindist)
        # if ev > 100:
        #     print("ALERT")


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
    print("error ITW :")
    print(valerr/len(allerror))




def dataPreparation_FD():
    # path = "/home/jonathanlo/Documents/DataMining/DM_Final_Project/Sofamehack2019/Sofamehack2019/Sub_DB_Checked/"+pathology+"/*"
    # filesCP = glob.glob(path) # Pour avoir tous les noms de fichiers dans une liste
    # X = []
    # Y = []
    # third = int(round(len(filesCP)/3))*2 # Deux tiers du nombre de fichiers

    for f in range(0,third):
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
        # LeftKnee = acq.GetPoint("LKNE").GetValues()
        # RightKnee = acq.GetPoint("RKNE").GetValues()
        # LeftThigh = acq.GetPoint("LTHI").GetValues()
        # RightThigh = acq.GetPoint("RTHI").GetValues()
        LeftTib = acq.GetPoint("LTIB").GetValues()
        RightTib = acq.GetPoint("RTIB").GetValues()
        for i in range(len(LeftHeel)): # Nous n'utilisons pas la profondeur de la marche
            LeftHeel[i][1] = 10
            LeftAnkle[i][1] = 10
            LeftToe[i][1] = 10
            # LeftKnee[i][1] = 10
            # LeftThigh[i][1] = 10
            LeftTib[i][1] = 10
            RightHeel[i][1] = 40
            RightAnkle[i][1] = 40
            RightToe[i][1] = 40
            # RightKnee[i][1] = 40
            # RightThigh[i][1] = 40
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
            # LKnee = LeftKnee[eventFr-1,:]
            # RKnee = RightKnee[eventFr-1,:]
            # LThigh = LeftThigh[eventFr-1,:]
            # RThigh = RightThigh[eventFr-1,:]
            LTib = LeftTib[eventFr-1,:]
            RTib = RightTib[eventFr-1,:]
            allValues = [] # Stocke les frames des evenements
            allValues.extend(LHeel)
            allValues.extend(RHeel)
            allValues.extend(LAnkle)
            allValues.extend(RAnkle)
            allValues.extend(LToe)
            allValues.extend(RToe)
            # allValues.extend(LKnee)
            # allValues.extend(RKnee)
            # allValues.extend(LThigh)
            # allValues.extend(RThigh)
            allValues.extend(LTib)
            allValues.extend(RTib)
            X.append(allValues) # Les frames des evenements
            Y.append(label + " " + context) # Les labels des evenements (les labels correspondants)



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
                # LKnee = LeftKnee[event + val,:]
                # RKnee = RightKnee[event + val,:]
                # LThigh = LeftThigh[event + val,:]
                # RThigh = RightThigh[event + val,:]
                LTib = LeftTib[eventFr + val,:]
                RTib = RightTib[eventFr + val,:]
                allValues = []
                allValues.extend(LHeel)
                allValues.extend(RHeel)
                allValues.extend(LAnkle)
                allValues.extend(RAnkle)
                allValues.extend(LToe)
                allValues.extend(RToe)
                # allValues.extend(LKnee)
                # allValues.extend(RKnee)
                # allValues.extend(LThigh)
                # allValues.extend(RThigh)
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
    for f in range(third,len(filesCP)):
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
        # LeftKnee = acq.GetPoint("LKNE").GetValues()
        # RightKnee = acq.GetPoint("RKNE").GetValues()
        # LeftThigh = acq.GetPoint("LTHI").GetValues()
        # RightThigh = acq.GetPoint("RTHI").GetValues()
        LeftTib = acq.GetPoint("LTIB").GetValues()
        RightTib = acq.GetPoint("RTIB").GetValues()
        for i in range(len(LeftHeel)):
            LeftHeel[i][1] = 10
            LeftAnkle[i][1] = 10
            LeftToe[i][1] = 10
            # LeftKnee[i][1] = 10
            # LeftThigh[i][1] = 10
            LeftTib[i][1] = 10
            RightHeel[i][1] = 40
            RightAnkle[i][1] = 40
            RightToe[i][1] = 40
            # RightKnee[i][1] = 40
            # RightThigh[i][1] = 40
            RightTib[i][1] = 40

        prediction = [[]]*(nbFrame-1)

        for i in range(1,nbFrame):
            LHeel = LeftHeel[i-1,:]
            RHeel = RightHeel[i-1,:]
            LAnkle = LeftAnkle[i-1,:]
            RAnkle = RightAnkle[i-1,:]
            LToe = LeftToe[i-1,:]
            RToe = RightToe[i-1,:]
            # LKnee = LeftKnee[i-1,:]
            # RKnee = RightKnee[i-1,:]
            # LThigh = LeftThigh[i-1,:]
            # RThigh = RightThigh[i-1,:]
            LTib = LeftTib[i-1,:]
            RTib = RightTib[i-1,:]
            allValues = []
            allValues.extend(LHeel)
            allValues.extend(RHeel)
            allValues.extend(LAnkle)
            allValues.extend(RAnkle)
            allValues.extend(LToe)
            allValues.extend(RToe)
            # allValues.extend(LKnee)
            # allValues.extend(RKnee)
            # allValues.extend(LThigh)
            # allValues.extend(RThigh)
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
        #writer.SetFilename('/home/guillaume/Bureau/PROJET DATA MINING OPERATIONAL/Results/CP/' + fname)
        writer.SetFilename(pathSave + fname)
        writer.Update()

        reader = btk.btkAcquisitionFileReader()
        #reader.SetFilename('/home/guillaume/Bureau/PROJET DATA MINING OPERATIONAL/Results/CP/' + fname)
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



        # print(leventFr)
        # print(llabel)
        # print(lcontext)


        # for i in range(0,len[levent[d]]):
        #     event = acq.GetEvent(i)
        #     eventFr = event.GetFrame() # return the frame as an integer
        #     leventFr.append(eventFr)
        #     label = event.GetLabel() # return a string representing the Label
        #     context = event.GetContext() # return a string representing the Context
        #     llabel.append(label)
        #     lcontext.append(context)
        # for o in range(0,len(leventFr)):
        #     mindist = nbFrame
        #     for p in range(0,len(lnewevent)):
        #         if lnel[p] == llabel[o] and lnec[p] == lcontext[o]:
        #             if abs(leventFr[o] - lnewevent[p]) < mindist :
        #                 mindist = abs(leventFr[o] - lnewevent[p])
        #     lerror.append(mindist)
        # if ev > 100:
        #     print("ALERT")


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
    # print(allerror)
    valerr = 0
    for k in allerror:
        for p in k:
            if p > 1:
                valerr = valerr + np.exp(p)

    print(" ")
    print("error FD :")
    print(valerr/len(allerror))
    print("")










###############MAIN
dataPreparation_CP()
prediction_CP()
path = "/home/jonathanlo/Documents/DataMining/DM_Final_Project/Sofamehack2019/Sofamehack2019/Sub_DB_Checked/ITW/*"
files = glob.glob(path)
X = []
Y = []
third = int(round(len(files)/3))*2

dataPreparation_ITW()
prediction_ITW()
path = "/home/jonathanlo/Documents/DataMining/DM_Final_Project/Sofamehack2019/Sofamehack2019/Sub_DB_Checked/FD/*"
files = glob.glob(path)
X = []
Y = []
third = int(round(len(files)/3))*2
dataPreparation_FD()
prediction_FD()
