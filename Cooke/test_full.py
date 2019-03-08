
"""
Copyright 2018, J. Cooke, Radboud University Nijmegen

AdaptiveModelSelection is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

AdaptiveModelSelection is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with AdaptiveModelSelection. If not, see <http://www.gnu.org/licenses/>.

---

some short testing functions

References:
Cooke, J. R. H., van Beers, R. J., Selen, L. P. J., & Medendorp, W. P.
Bayesian adaptive stimuli selection for dissociating psychophysical models.

"""
import time as ti
import numpy as np
import AdaptiveModelSelection as MS

# set constants to generate data
np.random.seed(7)

psyFunctions=('fixGauss') #functions and models to consider
sampleMethod=(['Rand']) # algorithm names
nTrials = 1000# n simulated trials
stimRange1=np.arange(0.05,0.4,0.01) #stimuli space
stimRange2= np.arange(0.05,0.4,0.03)
thresholdGrid= np.linspace(0,0,1)
slopeGrid = np.linspace(0.05, 0.6, 20)
lapseRateGrid = np.linspace(0, 0.15, 20)
gammaGrid = np.linspace(0.1, 3, 20)

thresholdPrior=('uniform',-0.1,0.5)
slopePrior=('uniform',np.min(slopeGrid),np.max(slopeGrid))
lapsePrior=('uniform',np.min(lapseRateGrid),np.max(lapseRateGrid))
gammaPrior=('uniform',np.min(gammaGrid),np.max(gammaGrid))

mprior=1/float(3)
fixgauss = MS.build_model(psyFunctions[0], mprior, ('alpha', 'beta', 'lapse'),
                          (thresholdPrior, slopePrior, lapsePrior),
                          (thresholdGrid, slopeGrid, lapseRateGrid))
webergauss = MS.build_model(psyFunctions[1], mprior, ('alpha', 'beta', 'lapse'),
                            (thresholdPrior, slopePrior, lapsePrior),
                            (thresholdGrid, slopeGrid, lapseRateGrid))
genWebGauss = MS.build_model(psyFunctions[2], mprior, ('alpha', 'beta', 'lapse', 'gamma'),
                             (thresholdPrior, slopePrior, lapsePrior, gammaPrior),
                             (thresholdGrid, slopeGrid, lapseRateGrid, gammaGrid))
models={'fixGauss':fixgauss,'weberGauss':webergauss,'genWebGauss':genWebGauss}

##set up data storage
response=np.empty(nTrials)
stim1=np.empty(nTrials)
stim2=np.empty(nTrials)
probdic={}
for mod in models:
    probdic[mod]=np.empty(nTrials)

thresholdGen=0
slopeGen = 0.5
lapseGen = 0
gamma    = 0.5

obj=MS.Psi(stimRange1, stimRange2, models, sampleMethod='Rand') #set up object (probably a better way to avoid recomputing Marginal likelihood)
stimuli=obj.xCurrent
stimuli2=obj.xCurrent2
##############################

print "\n##############################\n"
print "Simulating dataset\n"
print "Data generated from with following parameters\n"
print "Psychometric function: %s " %(psyFunctions)
print "Threshold: %.3f " %(thresholdGen)
print "Slope: %.3f " %(slopeGen)
print "Lapse Rate: %.3f " %(lapseGen)
print "Base Noise: %.3f " %(gamma)
print "\n##############################\n"
for mod in obj.prior:
    probdic[mod][0] = obj.prior[mod]['modelPrior']
timeStart=ti.time()
generativeParams={}
generativeParams = {'stim1': obj.xCurrent, 'stim2': obj.xCurrent2,
                    'alpha': thresholdGen, 'beta': slopeGen, 'lapse': lapseGen, 'gamma': gamma}

for i in range(0,nTrials):  # run for length of trials
    stim1[i]=stimuli
    stim2[i]=stimuli2
    generativeParams['stim1'] = stim1[i]
    generativeParams['stim2'] = stim2[i]
    generativeParams['beta'] = slopeGen * 5

    r = int(MS.genDat(parameters=generativeParams, psyfun=psyFunctions))  # generate simulated response
    response[i] = r
    obj.addData(r)  # update Psi with response
    for mod in obj.prior:
        probdic[mod][i] = obj.prior[mod]['modelPrior']
    stimuli=obj.xCurrent  # fetch new stimuli
    stimuli2=obj.xCurrent2
print "\n##############################\n"
print "Time elapsed: %s\n" %(ti.time()-timeStart)
print "Data estimated with following parameters\n"
recursiveEstimate=np.empty(len(obj.prior))
i=0
for mod in obj.prior:
    print "Psychometric function: %s : probability %s " %(mod,obj.prior[mod]['modelPrior'])
    recursiveEstimate[i]=obj.prior[mod]['modelPrior']
    i+=1
print "\n##############################\n"

modelProb=obj.batchmodelComp(stim1,stim2,response) #pass same data to batch model fit to check the results
batchEstimate=np.empty(len(obj.prior))
i=0
for mod in obj.prior:
    batchEstimate[i]=modelProb[mod]['ModelProbability']
    i+=1
print "\n##############################\n"

def test_recursive_against_batch():
    import numpy as np
    np.testing.assert_almost_equal(recursiveEstimate,batchEstimate,decimal=10)==1

def test_build_model():
    slopeGrid = np.linspace(0.05, 0.6, 20)
    lapseRateGrid = np.linspace(0, 0.15, 20)
    gammaGrid = np.linspace(0.1, 3, 20)

    slopePrior = ('uniform', np.min(slopeGrid), np.max(slopeGrid))
    lapsePrior = ('uniform', np.min(lapseRateGrid), np.max(lapseRateGrid))
    gammaPrior = ('uniform', np.min(gammaGrid), np.max(gammaGrid))

    mprior = 1 / float(3)
    fixgauss = MS.build_model('fixgauss', mprior, ('beta', 'lapse'), (slopePrior, lapsePrior),
                              (slopeGrid, lapseRateGrid))
    assert fixgauss['prior']==mprior
    assert np.array_equal(fixgauss['parameters']['beta']['searchGrid'],slopeGrid)

def test_generate_date():
    import pytest
    generativeParams = {'stim1': 1, 'stim2': 1,
                        'alpha': 1, 'beta': 1, 'lapse': 0, 'gamma': 0}
    generativeParams['stim1'] = stim1[i]
    generativeParams['stim2'] = stim2[i]
    psyFun='invalid'
    with pytest.raises(ValueError):
        MS.genDat(generativeParams, psyfun=psyFun) #invalid function raise error
    psyFun='fixGauss'
    r= MS.genDat(generativeParams, psyfun=psyFun)
    assert (r == 1) | (r == 0) #check we get a bernoulli variable out

def test_update_double_methods():
    obj = MS.Psi(stimRange1, stimRange2, models,
                 sampleMethod='Rand')  # set up object (probably a better way to avoid recomputing Marginal likelihood)
    obj2 = MS.Psi(stimRange1, stimRange2, models,
                 sampleMethod='AMS')  # set up object (probably a better way to avoid recomputing Marginal likelihood)

    assert obj.prior['fixGauss']['modelPrior'] == obj2.prior['fixGauss']['modelPrior']
    #prior probabilities should be equal (can't compare objects as marginal likelihood shapes are different)

    obj.addData(int(1))

    assert obj.prior['fixGauss']['modelPrior'] != obj2.prior['fixGauss']['modelPrior'] #updating should be independent between objects

def test_sensorynoise_likelihood():
    s1=10 #stim 1
    s2=13 #stim 2
    ws=0.2 #weber fraction
    w=1 #base noise level
    alpha=0 # bias (currently only programmed for alpha=0)
    lapse=0 # lapse
    generativeParams = {'stim1': s1, 'stim2': s2,
                        'alpha': alpha, 'beta': ws, 'lapse': lapse, 'gamma': w}
    psyFunctions=('fixGauss','weberGauss','genWebGauss')
    likAna=np.empty(len(psyFunctions))

    for i in range(0,len(psyFunctions)):
        if psyFunctions[i] == 'fixGauss':
            generativeParams['beta'] = ws * 5
        else:
            generativeParams['beta'] = ws
        likAna[i]=MS.Lik(generativeParams,psyFunctions[i])

    def sampleLikelihood(parms,psyfunction):
        if psyfunction=='fixGauss':
            parms['beta'] = ws*5
            x1=np.random.normal(parms['stim1'],parms['beta'],size=(30000))
            x2=np.random.normal(parms['stim2'],parms['beta'],size=(30000))
            Lik=np.mean((x1-x2)>0)
        elif psyfunction=='weberGauss':
            parms['beta'] = ws
            x1=np.random.normal(parms['stim1'],parms['beta']*s1,size=(30000))
            x2=np.random.normal(parms['stim2'],parms['beta']*s2,size=(30000))
            Lik = np.mean((x1 - x2) > 0)
        elif psyfunction=='genWebGauss':
            parms['beta'] = ws
            x1=np.random.normal(parms['stim1'],np.sqrt((parms['beta']*s1)**2+parms['gamma']**2),size=(30000))
            x2=np.random.normal(parms['stim2'],np.sqrt((parms['beta']*s2)**2+parms['gamma']**2),size=(30000))
            Lik = np.mean((x1 - x2) > 0)
        return Lik
    likSample=np.empty(len(psyFunctions))
    for i in range(0,len(psyFunctions)):
        likSample[i] = sampleLikelihood(generativeParams,psyFunctions[i])
    np.testing.assert_almost_equal(likSample, likAna, decimal=2) == 1


    # todo add a test for entropy computation (check einsum)
