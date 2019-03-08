# Anouk de Brouwer & Bart Alberts, March Adapted by Anneloes Ernest, June
#  Sensorimotorlab, Nijmegen
# Rod-and-frame experiment

import numpy, os
from psychopy import visual, core, data, event, gui, sys
from rusocsci import buttonbox

from PSIRiF import PSIfor

boolButtonBox = False

cwd = os.getcwd()

if boolButtonBox:
    bb = buttonbox.Buttonbox(port='COM2')

def waitResponse():
    global response
    key = ['']
    while key[0] not in ['z', 'm','escape']:
        key = event.waitKeys()
        if key[0] == 'z':
            response = 0
        elif key[0] == 'm':
            response = 1
        elif key[0] == 'escape':
            exit()

# transforms sigma values into kappa values
def sig2kap(sig): #in degrees
    sig2=numpy.square(sig)
    return 3.9945e3/(sig2+0.0226e3)


# experiment info
print('Possible conditions: practice, frame \nPossible locations: v(isionLab), a(noukTest)')
#expInfo = {'dayofbirth':'','expDate':data.getDateStr(),'subjectNr':'','nRepetitions':''}
expInfo = {'Subject Code':'','Year of Birth':''}

expInfoDlg = gui.DlgFromDict(dictionary=expInfo, title='Experiment details', fixed='expDate')
for key in expInfo.keys():
    assert not expInfo[key]=='', "Forgot to enter %s!"% key


# setup info
setupInfo = {'monitorResolution':[1920,1080],'monitorFrameRate':60,'monitorSize_cm':[122,67.5],'viewDistance_cm':57,'screenNr':1}
fileDir = cwd
fileName = 'RIF_{!s}_{!s}.txt'.format(expInfo['Year of Birth'],expInfo['Subject Code'])

# open a textfile and write the experiment and setup details to it
dataFile = open('{}'.format(fileName),'w')
dataFile.write('Rod-andframe experiment by Anneloes Ernest and Luc Selen, Sensorimotorlab Nijmegen \n')
dataFile.write('Rod-and-frame task: a rod-andframe stimulus is presented, and the participant has to indicate whether the rod is tilted counterclockwise (left arrow key, response=-1) or clockwise (right arrow key, response=1) with respect to the gravitational vertical.\n\n')

for key,value in expInfo.iteritems():
    dataFile.write('{} = {};\n'.format(key,value))

for key,value in setupInfo.iteritems():
    dataFile.write('{} = {};\n'.format(key,value))

dataFile.write('\n') # new line
dataFile.write('frameOri rodOri response reactionTime \n')
dataFile.close()

# create the window to draw in
resolution = setupInfo['monitorResolution']

#win = visual.Window(monitor="Philips", fullscr=True, units="cm", winType='pyglet', screen=1, color = 'black')
win=visual.Window(monitor="testMonitor", fullscr=True, units="cm", winType='pyglet', screen=0, color = 'black')

# stimulus colors
frameColor = (-.8,-.8,-.8)
rodColor = (-.8,-.8,-.8)

#frameColor = (0.0,0.0,0.0)
#rodColor = (0.0,0.0,0.0)


frame_size = 15
line_length = 6

# create a rod-and-frame stimulus
frame = visual.Rect(win,width=frame_size,height=frame_size,lineColor=frameColor,lineColorSpace='rgb',lineWidth=1,units = 'cm')
rod = visual.Line(win,start=(0, -line_length),end=(0,line_length),lineColor=rodColor,lineColorSpace='rgb',lineWidth=1,units = 'cm')

vert_center = -2
horz_center = +15.0


frame.pos = (horz_center, vert_center)
rod.pos = (horz_center, vert_center)

# stimulus orientations
# frameOri = range(-45,45,5)
frameOri = numpy.linspace(-45,45,11)
rodOri = numpy.linspace(-10,10,30)


# wait for a key press to start the experiment

startText1 = visual.TextStim(win,text='Press the left arrow if you believe that\n the line is tilted counterclockwise.\nPress the right arrow if you believe the that\n the line is tilted clockwise.',pos=(horz_center,vert_center+2),alignHoriz='center',color=rodColor,wrapWidth=25, height =1)
startText2 = visual.TextStim(win,text='Ready?\n Press a button to start.',alignHoriz ='center',pos=(horz_center,vert_center-5),color=rodColor, height = 1)


startText1.draw()
startText2.draw()
win.flip()


if boolButtonBox:
    b = bb.waitButtons()

else:
    event.waitKeys(maxWait=60)
core.wait(1.0)

# experiment: present stimulus and wait for keyboard response
n = 0
t0=core.getTime

#init parameter ranges
#lookup9
kappa_oto = numpy.linspace(sig2kap(1.4),sig2kap(3.0),10)
kappa_ver = numpy.linspace(sig2kap(2.5),sig2kap(7.5),15)
kappa_hor = numpy.linspace(sig2kap(22),sig2kap(80),15)
tau = numpy.linspace(0.6,1.0,10);

#init algorithm
psi=PSIfor(kappa_ver,kappa_hor,tau,kappa_oto,frameOri,rodOri)

for trial in range(0,500):
    while psi.stim == None:
        pass

    # set rod and frame orientations
    stim_frame=psi.stim[0]
    stim_rod=psi.stim[1]

    frame.setOri(stim_frame)
    rod.setOri(stim_rod)

    # draw and flip only the frame
    frame.draw()
    win.flip()
    core.wait(.25) # time that the frame is visible

    # add the rod for 1 frame
    frame.draw()
    rod.draw()
    win.flip()
    # add the rod for a 2nd frame
    frame.draw()
    rod.draw()
    win.flip()

    # draw and flip only the frame
    frame.draw()

    win.flip()
    timer1 = core.getTime()

    # wait for a key press and then remove the frame
    if boolButtonBox:
        b = bb.waitButtons()
        key = b[0]
    else:
        waitResponse()
        #roughRT = keypress[0][1] - timer1
    if 'escape' in event.getKeys():
        exit()
    else:
        # get response from buttonbox
        if boolButtonBox:
            if 'A' == key:
                response = 0
            elif 'B' == key:
                response = 1
            else:
                response = 99
        roughRT = core.getTime()-timer1
        #write data to text file
        with open('{}'.format(fileName),'a') as dataFile:
            dataFile.write('{:1.1f} {:1.1f} {:1.1f} {:1.2f}\n'.format(stim_frame,stim_rod,response    ,roughRT))
        win.flip()
        #update priors
        params = psi.addData(response)
        print trial, stim_frame, stim_rod, response
        psi.print_expected_value()
        core.wait(0.2) # intertrial interval, black screen


ExpDur = core.getTime()-t0
event.waitKeys(maxWait=60)
# close
win.close()
core.quit()