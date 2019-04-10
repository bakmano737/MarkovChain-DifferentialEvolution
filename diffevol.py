##########################
# diffevol.py            #
# Differential Evolution #
#   by James V. Soukup   #
#   for CEE 290 HW #4    #
##########################
##################################################
# The models and cost functions are in models.py #
##################################################
import numpy as np
from numpy import random as rnd

######################################################################
# Differential Evolution                                             #
######################################################################
#  Recombination:                                                    #
#    child = parent + fde*(mate1-mate2)                              #
#    mate1 - First randomly selected member of the population        #
#    mate2 - Second randomly selected member of the population       #
#  Parameters:                                                       #
#    Pop    - Initial population of parameters                       #
#    cost   - Costs of initial population                            #
#    cr     - Crossover probability                                  #
#    fde    - Child variability factor                               #
#    pmut   - Mutation Probability                                   #
#    i      - Generation counter                                     #
#    im     - Max Generation Count                                   #
#    etol   - Exit Tolerance (Convergance)                           #
#    hist   - Lowest SSR of all previous generations (Analysis)      #
#    cf     - Cost Function                                          #
#    carg   - Cost Function Arguments                                #
######################################################################
def diffevol(Pop,cost,cr,fde,pmut,i,im,hist,etol,cf,carg):
    #########################
    # Step One: Selection   #
    #########################
    # Generate two unique random integers #
    # for each member of the population   #
    r = rnd.choice(Pop[:,0].size, (Pop[:,0].size,2))
    # Replace pairs of duplicates with a unique pair
    dup    = r[:,0]==r[:,1]
    r[dup] = rnd.choice(Pop[:,0].size,r[dup].shape,False)
    # Define the mating partners
    FirstMates = Pop[r[:,0],:]
    SecndMates = Pop[r[:,1],:]
    ####################
    # Step Two: Mating #
    ####################
    # Partial Crossover
    Pcr = rnd.choice([0,1],Pop.shape,p=[1-cr,cr])
    # Recombination
    mateDiff = np.subtract(FirstMates,SecndMates)
    crssover = np.multiply(fde*Pcr,mateDiff)
    Child    = np.mod(np.add(Pop,crssover),1)
    # Mutation
    Mut = rnd.rand(*Child.shape)
    Mut = Mut<pmut
    Child[Mut] = rnd.rand(*Child[Mut].shape)
    #########################
    # Step Three: Rejection #
    #########################
    # Evaluate Cost for Child Population
    chCst = cf(Child,carg)
    costc = chCst[1][1]
    costp = cost[1][1]
    # Replace dominated offspring with parent
    dom = np.array(np.greater(costc,costp)).reshape((-1,))
    Child[dom] = Pop[dom]
    np.minimum(costc,costp,out=costc)
    chCst[1][1] = costc

    # Best in show
    best = np.min(costc)
    hist[i] = best

    # Check convergance
    #if best <= etol:
    #   return [Child,chCst]

    # Check Generation Counter 
    if (im <= i+1):
        # Maximum Number of generations reached
        # Return the current population
        return [Child,chCst]

    ##############################
    # Create the next generation #
    ##############################
    return diffevol(Child,chCst,cr,fde,pmut,i+1,im,hist,etol,cf,carg)

######################################################################
# Differential Evolution Alternate Recombination                     #
######################################################################
#  Recombination:                                                    #
#    child = parent + fde*(mate1-mate2) + lam*(best-parent)          #
#    best  - Individual with lowest SSR in current generation        #
#    mate1 - First randomly selected member of the population        #
#    mate2 - Second randomly selected member of the population       #
#  Parameters:                                                       #
#    Pop   - Initial population of parameters                        #
#    cost  - Costs of initial population                             #
#    cr    - Crossover probability                                   #
#    fde   - Child variability factor                                #
#    lam   - Best parent scaling factor                              #
#    pmut  - Mutation Probability                                    #
#    i     - Generation counter                                      #
#    im    - Max Generation Count                                    #
#    etol  - Exit Tolerance (Convergance)                            #
#    hist  - Lowest SSR of all previous generations (Analysis)       #
#    cf    - Cost Function                                           #
#    carg  - Cost Function Arguments                                 #
######################################################################
def dealt(Pop,cost,cr,fde,lam,pmut,i,im,hist,etol,cf,carg):
    #########################
    # Step One: Selection   #
    #########################
    # Generate two unique random integers #
    # for each member of the population   #
    r = rnd.choice(Pop[:,0].size, (Pop[:,0].size,2))
    # Replace pairs of duplicates with a unique pair
    dup    = r[:,0]==r[:,1]
    r[dup] = rnd.choice(Pop[:,0].size,r[dup].shape,False)
    # Define the mating partners
    FirstMates = Pop[r[:,0],:]
    SecndMates = Pop[r[:,1],:]
    # Best in show
    besti = np.argmin(cost[1][1])
    bestp = Pop[besti,:]
    hist[i] = cost[1][1][besti]

    ####################
    # Step Two: Mating #
    ####################
    # Partial Crossover
    Pcr = rnd.choice([0,1],Pop.shape,p=[1-cr,cr])
    # Recombination
    mateDiff = np.subtract(FirstMates,SecndMates)
    bestDiff = np.subtract(bestp,Pop)
    crssover = np.multiply(fde*Pcr,mateDiff)
    bestover = np.multiply(lam*Pcr,bestDiff)
    fullover = np.add(crssover,bestover)
    Child    = np.mod(np.add(Pop,fullover),1)
    # Mutation
    Mut = rnd.rand(*Child.shape)
    Mut = Mut<pmut
    Child[Mut] = rnd.rand(*Child[Mut].shape)
    #########################
    # Step Three: Rejection #
    #########################
    # Evaluate Cost for Child Population
    chCst = cf(Child,carg)
    costc = chCst[1][1]
    costp = cost[1][1]
    # Replace dominated offspring with parent
    dom = np.array(np.greater(costc,costp)).reshape((-1,))
    Child[dom] = Pop[dom]
    np.minimum(costc,costp,out=costc)
    chCst[1][1] = costc

    # Check convergance
    #if best <= etol:
    #   return [Child,chCst]

    # Check Generation Counter 
    if (im <= i+1):
        # Maximum Number of generations reached
        # Return the current population
        return [Child,chCst]

    ##############################
    # Create the next generation #
    ##############################
    return dealt(Child,chCst,cr,fde,lam,pmut,i+1,im,hist,etol,cf,carg)

######################################################################
# Differential Evolution Multi Objective Pareto Ranking              #
######################################################################
#  Recombination:                                                    #
#  Parameters:                                                       #
#    Pop   - Initial population of parameters                        #
#    cost  - Costs of initial population                             #
#    cr    - Crossover probability                                   #
#    fde   - Child variability factor                                #
#    lam   - Best parent scaling factor                              #
#    pmut  - Mutation Probability                                    #
#    bhs   - Boundary Handling Strategy (0-wrap,1-reflect,2-snap)    #
#    i     - Generation counter                                      #
#    im    - Max Generation Count                                    #
#    cf    - Cost Functions                                          #
######################################################################
def demo(Pop,Cost,cr,fde,pmut,bhs,i,im,cf):
    #########################
    # Step One: Selection   #
    #########################
    # Generate two unique random integers #
    # for each member of the population   #
    N = Pop.shape[0]
    p = Pop.shape[1]
    c = Cost.shape[1]
    r = rnd.choice(N, (N,2))
    # Replace pairs of duplicates with a unique pair
    dup    = r[:,0]==r[:,1]
    r[dup] = rnd.choice(N,r[dup].shape,False)
    # Neither element of r can be its own index
    a = np.arange(N).reshape((N,1))
    r[np.equal(r,a)] = np.mod(r[np.equal(r,a)]+1,N)
    # Define the mating partners
    FirstMates = Pop[r[:,0],:]
    SecndMates = Pop[r[:,1],:]
    ####################
    # Step Two: Mating #
    ####################
    # Partial Crossover
    Pcr = rnd.choice([0,1],Pop.shape,p=[1-cr,cr])
    # Recombination
    mateDiff = np.subtract(FirstMates,SecndMates)
    crssover = np.multiply(fde*Pcr,mateDiff)
    newchild = np.add(Pop,crssover) 
    # Maintain Parameter Space
    if bhs == 0:
        # Wrap-around
        Child = np.mod(newchild,1)
    elif bhs ==1:
        # Reflection
        Child = np.mod(-newchild,1)
    elif bhs ==2:
        # Set to Bound
        Child = newchild
        Child[Child>1] = 1.0
        Child[Child<0] = 0.0
    else:
        print("Invalid BHS")
        exit()
    # Mutation
    Mut = rnd.rand(*Child.shape)
    Mut = Mut<pmut
    Child[Mut] = rnd.rand(*Child[Mut].shape)

    #########################
    # Step Three: Rejection #
    #########################
    # Evaluate Cost for Child Population
    ChCst = cf(Child)
    # Pareto Ranking
    # Gather the info for the population
    Parents  = np.hstack((Pop,Cost))
    Children = np.hstack((Child,ChCst))
    TotalPop = np.vstack((Parents, Children))
    # Rank the combined population
    Ranked = compRank(TotalPop,p,c)
    Parents = Ranked[:N,:]
    Children = Ranked[N:,:]
    mask = Children[:,0]>Parents[:,0]
    Children[mask] = Parents[mask]
    # Check Generation Counter 
    if (im <= i+1):
        # Maximum Number of generations reached
        # Return the rank1 population and cost
        return Children

    ##############################
    # Create the next generation #
    ##############################
    Child = Children[:,1:p+1]
    ChCst = Children[:,p+1:]
    return demo(Child,ChCst,cr,fde,pmut,bhs,i+1,im,cf)


######################################################################
# Differential Evolution Multi Objective Pareto Ranking w/Matlab     #
######################################################################
#  Recombination:                                                    #
#  Parameters:                                                       #
#    Pop   - Initial population of parameters                        #
#    cost  - Costs of initial population                             #
#    cr    - Crossover probability                                   #
#    fde   - Child variability factor                                #
#    lam   - Best parent scaling factor                              #
#    pmut  - Mutation Probability                                    #
#    bhs   - Boundary Handling Strategy (0-wrap,1-reflect,2-snap)    #
#    i     - Generation counter                                      #
#    im    - Max Generation Count                                    #
#    cf    - Cost Functions                                          #
#    mleng = Matlab Engine                                           #
######################################################################
def demo_ml(Pop,Cost,cr,fde,pmut,bhs,i,im,cf,mleng):
    #########################
    # Step One: Selection   #
    #########################
    # Generate two unique random integers #
    # for each member of the population   #
    N = Pop.shape[0]
    p = Pop.shape[1]
    c = Cost.shape[1]
    r = rnd.choice(N, (N,2))
    # Replace pairs of duplicates with a unique pair
    dup    = r[:,0]==r[:,1]
    r[dup] = rnd.choice(N,r[dup].shape,False)
    # Neither element of r can be its own index
    a = np.arange(N).reshape((N,1))
    r[np.equal(r,a)] = np.mod(r[np.equal(r,a)]+1,N)
    # Define the mating partners
    FirstMates = Pop[r[:,0],:]
    SecndMates = Pop[r[:,1],:]

    ####################
    # Step Two: Mating #
    ####################
    # Partial Crossover
    Pcr = rnd.choice([0,1],Pop.shape,p=[1-cr,cr])
    # Recombination
    mateDiff = np.subtract(FirstMates,SecndMates)
    crssover = np.multiply(fde*Pcr,mateDiff)
    newchild = np.add(Pop,crssover) 
    # Maintain Parameter Space
    if bhs == 0:
        # Wrap-around
        Child = np.mod(newchild,1)
    elif bhs ==1:
        # Reflection
        Child = np.mod(-newchild,1)
    elif bhs ==2:
        # Set to Bound
        Child = newchild
        Child[Child>1] = 1.0
        Child[Child<0] = 0.0
    else:
        print("Invalid BHS")
        exit()
    # Mutation
    Mut = rnd.rand(*Child.shape)
    Mut = Mut<pmut
    Child[Mut] = rnd.rand(*Child[Mut].shape)

    #########################
    # Step Three: Rejection #
    #########################
    # Evaluate Cost for Child Population
    # Start the Matlab Engine
    ChCst = cf(Child)
    # Pareto Ranking
    # Gather the info for the population
    Parents  = np.hstack((Pop,Cost))
    Children = np.hstack((Child,ChCst))
    TotalPop = np.vstack((Parents, Children))
    # Rank the combined population
    Ranked = compRank(TotalPop,p,c)
    Parents = Ranked[:N,:]
    Children = Ranked[N:,:]
    mask = Children[:,0]>Parents[:,0]
    Children[mask] = Parents[mask]
    # Check Generation Counter 
    if (im <= i+1):
        # Maximum Number of generations reached
        # Return the rank1 population and cost
        return Children

    ##############################
    # Create the next generation #
    ##############################
    Child = Children[:,1:p+1]
    ChCst = Children[:,p+1:]
    return demo_ml(Child,ChCst,cr,fde,pmut,bhs,i+1,im,cf,mleng)

######################################################################
# Differenctial Evolution with Metropolis Acceptance Rule            #
######################################################################
#  Parameters:                                                       #
#    Pop   - Initial population of parameters                        #
#    Cost  - Costs of initial population                             #
#    cr    - Crossover probability                                   #
#    gam   - Child variability factor                                #
#    pmut  - Mutation Probability                                    #
#    bhs   - Boundary Handling Strategy (0-wrap,1-reflect,2-snap)    #
#    T     - Max Generation Count (Number of chain states)           #
#    cf    - Cost Functions                                          #
#    cargs - Cost Function arguments                                 #
######################################################################
def de_mc(Pop,Cost,cr,gam,pmut,bhs,T,cf,cargs):
    # Relevant Population Sizes
    # Number of Chains
    N = Pop.shape[0]
    # Number of Parameters
    p = Pop.shape[1]
    # Number of chain-states
    T = Pop.shape[2]
    # Number of cost-functions
    c = Cost.shape[1]
    # Generation Shape
    gs = (N,p)
    for t in range(T-1):
        #########################
        # Step One: Selection   #
        #########################
        # Generate two unique random integers #
        # for each member of the population   #
        r = rnd.choice(N,(N,2))
        # Replace pairs of duplicates with a unique pair
        dup = r[:,0]==r[:,1]
        r[dup,1] = np.mod(r[dup,1]+rnd.choice(np.arange(1,N)),N)
        # Neither element of r can be its own index
        a = np.arange(N).reshape((N,1))
        r[np.equal(r,a)] = np.mod(r[np.equal(r,a)]+1,N)
        # Define the mating partners
        FirstMates = Pop[r[:,0],:,t]
        SecndMates = Pop[r[:,1],:,t]
        ####################
        # Step Two: Mating #
        ####################
        # Partial Crossover
        Pcr = rnd.choice([0,1],gs,p=[1-cr,cr])
        ed  = rnd.normal(0,1e-6,N*p).reshape(gs)
        # Recombination
        mateDiff = np.subtract(FirstMates,SecndMates)
        crssover = np.multiply(gam*Pcr,mateDiff)
        errchild = np.add(Pop[:,:,t],crssover)
        newchild = np.add(errchild,ed)
        # Maintain Parameter Space
        if bhs == 0:
            # Wrap-around
            Child = np.mod(newchild,1)
        elif bhs == 1:
            # Reflection
            Child = np.mod(-newchild,1)
        elif bhs == 2:
            # Set to Bound
            Child = newchild
            Child[Child>1] = 1.0
            Child[Child<0] = 0.0
        else:
            print("Invalid BHS")
            exit()
        # Mutation
        Mut = rnd.rand(*Child.shape)
        Mut = Mut<pmut
        Child[Mut] = rnd.rand(*Child[Mut].shape)
        #########################
        # Step Three: Rejection #
        #########################
        # Evaluate Cost for Child Population
        ChCst = cf(Child,*cargs)
        # Determine the Metropolis Ratio (do not divide by 0!)
        kwa = {'out':np.zeros_like(ChCst),'where':Cost[:,:,t]!=0}
        alp = np.divide(ChCst,Cost[:,:,t],**kwa)
        # No acceptance rates greater than 1
        alp = np.minimum(1,alp)
        # Random draw from U(0,1) for acceptance probability
        pac = np.random.rand(*alp.shape)
        # Determine acceptance
        acc = np.greater_equal(alp,pac)
        # Add the accepted children to the new chain state
        Pop[acc,t+1]  = Child[acc]
        # Add the parent where the children were rejected
        Pop[~acc,t+1] = Pop[~acc,t]
        # And the cost to the corresponding cost-chain
        Cost[acc,t+1]  = ChCst[acc]
        Cost[~acc,t+1] = Cost[~acc,t]
    return [Pop,Cost]

def bestRank(Cost):
    Pareto = np.ones(Cost.shape[0],dtype=bool)
    for i,cst in enumerate(Cost):
        if Pareto[i]:
            Pareto[Pareto] = ~np.all(Cost[Pareto]>=cst,axis=1)
            # The above identifies the current point as innefficient
            # all of it's costs are equal to it's own costs
            # This point may still be optimal
            Pareto[i] = True
    return Pareto

def compRank(Pop,p,c):
    N = Pop.shape[0]
    rank = 1
    Rank = np.zeros((N,1))
    Ranked = np.empty((0,p+c+1))
    while Ranked.shape[0] < N:
        b = bestRank(Pop[:,p:])
        r = rank*np.ones((b.shape[0],1))
        ranked = np.hstack((r[b],Pop[b,:]))
        Ranked = np.vstack((Ranked,ranked))
        Pop = Pop[~b]
        Rank = Rank[~b]
        rank += 1
    return Ranked
