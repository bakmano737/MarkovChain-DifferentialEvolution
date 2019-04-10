############################################################
# Random Walk Metropolis #

import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import diffevol as de

# Square Root of 2*pi
_s2p_ = 2.5066

def rwm(xi,px,F,Fp,q,qp,tmax):
    # Iterate through the required chain length
    for t in range(tmax-1):
        # Get the step from the provided distribution
        dlt = q(*qp)
        # Create the proposal point
        xip = xi[t] + dlt
        # Determine the "fitness" of the proposal point
        pxp = F(xip,**Fp)
        # Determine the Metropolis Ratio (do not divide by 0!)
        kwa = {'out':np.zeros_like(pxp),'where':px[t]!=0}
        alp = np.divide(pxp,px[t],**kwa)
        # No acceptance rates greater than 1
        acc = np.minimum(1,alp)
        # Random draw from U(0,1) for acceptance probability
        pac = np.random.rand()
        # Determine acceptance
        if acc > pac:
            # High Metropolis Ratio - accept the proposal
            xi[t+1] = xip
            px[t+1] = pxp
        else:
            #  Low Metropolis Ratio - reject the proposal
            xi[t+1] = xi[t]
            px[t+1] = px[t]
    # Return the Chain and Fitness
    return [xi,px]

###########################################################
# Standard Normal Distribution
def snd(x0,m=0,d=1,dom=[-5.0,5.0]):
    x = denorm(x0,*dom)
    return (1/(d*_s2p_))*np.exp(-0.5*((x-m)/d)**2)

def sndTest():
    x = np.arange(-5.0,5.0,0.01)
    qp = []
    p = snd(x,*qp)
    c = np.cumsum(p)
    plt.plot(x,p,x,c)
    plt.show()

############################################################
# Bimodal Normal Distribution
def bnd(x0,m1=-5,d1=1,m2=5,d2=1,dom=[-10.0,10.0]):
    return snd(x0,m=m1,d=d1,dom=dom)/3 + 2*snd(x0,m=m2,d=d2,dom=dom)/3

def bndTest():
    dm = [-10.0,10.0]
    x0 = np.arange(0.0,1.0,0.001)
    xd = denorm(x0,*dm)
    
    bx = bnd(x0)
    cd = np.cumsum(bx*0.02)
    plt.plot(xd,bx,xd,cd)
    plt.show()

def norm(x,mn,mx):
    return (x-mn)/(mx-mn)

def denorm(x0,mn,mx):
    return (mx-mn)*x0+mn

def testBayes(dist,alg,dom):
    if alg == "rwm":
        # Allocate Variables
        T  = 10000
        xi = np.empty(T)
        pr = np.empty(T)
        # Chain State Initialization
        xi[0] = np.array([rnd.rand()])
        pr[0] = np.array(dist(xi[0]))
        xo,po = rwm(xi,pr,dist,{'dom':dom},rnd.uniform,[-0.1,0.1],T)
        # Remove Burn-in
        xon = denorm(xo[5000:],*dom)
        pon = po[5000:]
    elif alg == "de-mc":
        # Allocate Variables
        N  = 10
        T  = 2500
        p  = 1
        c  = 1
        xi = np.empty((N,p,T))
        pr = np.empty((N,c,T))
        # Chain State Initialization
        xi[:,:,0] = np.array([rnd.rand(N,p)])
        pr[:,:,0] = np.array(dist(xi[:,:,0]))
        chains = de.de_mc(xi,pr,0.7,0.7,0.25,0,T,dist,[])
        # Separate the Parameter and Cost Chains
        Pop = chains[0]
        Cst = chains[1]
        # Remove Burn-in
        FPop = Pop[:,:,500:]
        FCst = Cst[:,:,500:]
        # Combine the chains Fotran Style
        #xon = denorm(Pop.flatten('F')[5000:],*dom)
        #pon = Cst.flatten('F')[5000:]
        xon = denorm(FPop.flatten('F'),*dom)
        pon = FCst.flatten('F')
    else:
        print("OH GOD WHY!?!?!?")
        exit()
    # Define histogram bin edges
    bw = 0.25
    bns = np.arange(*dom,bw)
    # Histogram Plot Keyword Arguments
    kwa = {'density':'True','bins':bns,'label':'Hist','zorder':10}
    # Create and Plot the histogram
    histo = plt.hist(xon,**kwa)
    hist = histo[0]
    bins = histo[1]
    # Create the cumulative distribution
    cdf = np.cumsum(hist*bw)
    # Determine the 95%CI edges from CDF
    ci25  = np.interp(0.025,cdf,bins[:-1])
    ci97 = np.interp(0.975,cdf,bins[:-1])
    # Compute and Print Mean, Standard Deviation and 95CI
    mnst = "Mean:{0:4.2f}".format(np.mean(xon))
    sdst = "Std. Dev.:{0:4.2f}".format(np.std(xon))
    c2st = "CI  2.5%:{0:4.2f}".format(ci25)
    c9st = "CI 97.5%:{0:4.2f}".format(ci97)
    print(mnst)
    print(sdst)
    print(c2st)
    print(c9st)
    # Plot the simulated PDF and calculated CDF
    plt.plot(xon,      pon,'go',label="Sim PDF",zorder=0)
    plt.plot(bins[:-1],cdf,'r', label="CDF")
    # Finally, axis labels and legend
    lbl="Normalized Frequency (%)"
    plt.ylabel(lbl)
    plt.xlabel("x")
    plt.legend()
    plt.show()
    # Trace Plot
    tr = denorm(np.squeeze(Pop[0:5,:,:]),*dom)
    lbs = ["Chain 1","Chain 2","Chain 3","Chain 4","Chain 5"]
    plt.plot(tr.T,marker='+',ls='None')
    plt.legend(lbs,loc='upper right')
    plt.ylabel("X Value")
    plt.xlabel("Generation Number")
    plt.grid(True)
    plt.show()

#bndTest()
testBayes(bnd,"de-mc",[-10.0,10.0])