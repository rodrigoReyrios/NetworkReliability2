import itertools as it
import concurrent.futures
from scipy import special as sp
import numpy as npp
import re
from os import listdir
import pickle
import time
from scipy.special import comb as choose
from sage.graphs.graph_generators import graphs
from copy import deepcopy
import math

##############################################################################################
#Tiny Utility functions

def Diff(li1, li2):
    '''
    Simple Wrapper for Set Diffrence
    '''
    return (set(li1) - set(li2))

def find(lis,item):
    '''
    Wrapper for finding items in a python list using numpy array
    realy only should be used for 1D python list
    '''
    #turn lis into numpy array
    lis2 = npp.array(lis)
    #return indices of where items are located
    return npp.flatnonzero(lis2==item)

def Choose(n,m):
    '''
    Simple wrapper to wrap scipys choose function to return int
    '''
    return int(choose(n,m))

###############################################################################################
#File managment functions

def FindGG(dirs=None):
    '''
    Searches Path for the largest GG_n#.pkl file names
    If no directory is specified the active one is used by default
    '''

    #setup the pattern
    r = re.compile("GG_n([0-9]+).pkl")

    #get list of files
    if dirs == None:
        Files = listdir()
    else:
        Files = listdir(dirs)

    #locate the GG files
    newlist = list(filter(r.match, Files))

    if len(newlist)==0:
        raise ValueError('No GG files found in designated Directory')
    
    #look for largest n this will have the most components
    rsub = re.compile('[0-9]+')
    maxID = 0
    maxrunning = 0
    for i,fname in enumerate(newlist):
        #find the span of where the numbers occur
        beg,end = rsub.search(fname).span()
        num = int(fname[beg:end])
        if num>maxrunning:
            maxID = i
            maxrunning = num
    #return the file at maxID
    return newlist[maxID]

def FindCompLib(dirs = None,sym = 'Chi', typ = 'E'):
    '''
    Searches for a CompLib file, if dir is not specified just uses the directory where this file 
    is. The comp file location is specified by the Sym and Typ arguments
    '''
    #setup pattern to find CompLib
    r = re.compile(f"CompLib_n([0-9]+)_{sym}_{typ}")

    #get list of files based on dir
    if dir==None:
        Files = listdir()
    else:
        Files = listdir(dirs)

    #apply the CompLib patter
    newlist = list(filter(r.match, Files))

    #raise error if we get no matches
    if len(newlist)==0:
        raise ValueError('No CompLib files found in designated Directory')

    #Now that the matches are returned Im going to look through tem and return the largest
    #that is highest n
    rsub = re.compile('n[0-9]+')

    maxID = 0
    maxrunning = 0

    for i,fname in enumerate(newlist):
        #find the span of where the numbers occur
        beg,end = rsub.search(fname).span()
        num = int(fname[beg:end][1:])
        if num>maxrunning:
            maxID = i
            maxrunning = num
            
    #return the file at maxID
    return newlist[maxID]

    return newlist

def LoadGG(dirs=None,prints=False):
    '''
    Looks in dir specified path for a GG file and loads it
    '''
    #locate a GG file
    fname = FindGG(dirs)
    #if dirs is None I need an empty string if its something I need the dirs+"/"
    if dirs is None:
        path = ''
    else:
        path = dirs+"/"
    if prints:
        print(f'Found GG file: {fname}')
    #unpickle the file
    infile = open(path+fname,'rb')
    GraphsData = pickle.load(infile)
    infile.close()
    #return the unpickled file
    return GraphsData

def LoadCompLib(dirs=None,sym='Chi',typ='E',prints=False):
    '''
    Looks in dir specified path for a CompLib file and loads it
    Automaticly selects file with largest n
    '''
    #locate a GG file
    fname = FindCompLib(dirs=dirs,sym=sym,typ=typ)
    #set the path in string form
    if dirs is None:
        path = ''
    else:
        path = dirs+"/"

    if prints:
        print(f'Found CompLib file: {fname}')
    #unpickle the file
    infile = open(path+fname,'rb')
    GraphsData = pickle.load(infile)
    infile.close()
    #return the unpickled file
    return GraphsData

##################################################################################################
#Basic Graph Utility functions
def IsTrivial(G):
    '''
    Check if a graph is trivial i.e. a single vertex
    '''
    return (G.order()==1 or G.order==0)

def GraphSetCompare(G,Glist):
    '''
    Will check to see if a graph G is isomorphic to a key in the dictionary Glist
    Will either return the key that its isomorphic to or will return None
    '''
    for H in Glist.keys():
        if G.is_isomorphic(H):
            return H
    return None

def IDSingletons(H):
    '''
    Function to identify the location of singleton vertices in graphs
    
    '''
    #get the degree list of the vertices
    degs = H.degree()
    #just return [] if no zero degree vertices found
    if 0 in degs:
        return find(degs,0).tolist()
    else:
        return []

def SingletonTrim(G):
    '''
    Removes all the singltons from a graph in place
    '''
    #id the singletons in the graph
    singleID = IDSingletons(G)
    G.delete_vertices(singleID)

def RemSubGraph(G,sub,typ='E'):
    '''
    Gives the Subgraph of G with the sub removed, i.e. G-sub
    typ specifies if sub is a set of vertices or edges
    '''
    #make a copy
    H = G.copy(immutable=False)
    if typ=='E':
        #builtin edge removal for graphs
        H.delete_edges(sub)
    elif typ=='V':
        #built in vertex removal
        H.delete_vertices(sub)
    else:
        print(f'Type: {typ} not recognized')
        return None
    return H

def RemSubGraphInPlace(G,sub,typ='E'):
    '''
    Same as RemSubGraph but done in place, faster since this doeant need to copy a whole graph
    for returns (Although this makes this have less functionality)
    '''
    if typ=='E':
        #builtin edge removal for graphs
        G.delete_edges(sub)
    elif typ=='V':
        #built in vertex removal
        G.delete_vertices(sub)
    else:
        print(f'Type: {typ} not recognized')

##################################################################################################
#Precomputation functions

def task(tup):
    '''
    task for uses in GetGG multithreading function, for some reason this cant be in the 
    GenGG local scope
    '''
    n,m = tup
    #make the Gnm set
    Gnm = list(graphs(n,size=m))
    return [Gnm,tup]

def task2(pair):
    '''
    Copy of task it seems, i dont recall why i needed this
    '''
    n,m = pair
    return [m,list(graphs(n,size=m))]

def task3(key):
    '''
    Task to compute components under a certain amount of vertices for a type of k
    '''
    np,sym,typ,k,dirs,prints,prints2 = key
    #run the function
    CL = CompLibSingleK(np=np,sym=sym,typ=typ,k=k,dirs=dirs,prints=prints,prints2=prints2)
    return [k,CL]

def task4(G):
    '''
    Task to take ina graph and compute its components and return them in pairs (order,size,components)
    '''
    #trim the graph of singletons
    SingletonTrim(G)
    #split the graph among components
    ComponentsToAdd = []
    for H in G.connected_components_subgraphs():
        H = H.copy(immutable=True)
        #get the size and order of this component
        ni = H.order()
        mi = H.size()
        ComponentsToAdd.append([ni,mi,H])
    return ComponentsToAdd

def task5ChiE(key):
    '''
    Task meant to be used as a wrapper to ompute a components value for Chi_E^k
    also checks if graph is complete and does a shortcut if so
    '''
    k,ni,mi,H = key
    maxn = Choose(ni,2)
    if mi==maxn:
        ChiEkComplete(ni,k)
    elif mi==(maxn-1):
        ChiEkCompleteMinusOne(ni,k)
    #define the computation I want
    Comp = lambda H: MinMax(H,sym ='Chi',typ='E',k=k)
    return k,ni,mi,H,Comp(H)

def task5ChiV(key):
    '''
    Task meant to be used as a wrapper to ompute a components value for Chi_V^k
    also checks if graph is complete and does a shortcut if so
    '''
    k,ni,mi,H = key
    maxn = Choose(ni,2)
    if mi==maxn:
        ChiVkComplete(ni,k)
    elif mi==(maxn-1):
        ChiVkCompleteMinusOne(ni,k)
    #define the computation I want
    Comp = lambda H: MinMax(H,sym ='Chi',typ='V',k=k)
    return k,ni,mi,H,Comp(H)

def task5LmbE(key):
    '''
    Task meant to be used as a wrapper to ompute a components value for Lmb_E^k
    also checks if graph is complete and does a shortcut if so
    '''
    k,ni,mi,H = key
    maxn = Choose(ni,2)
    if mi==maxn:
        LmbEkComplete(ni,k)
    elif mi==(maxn-1):
        LmbEkCompleteMinusOne(ni,k)
    #define the computation I want
    Comp = lambda H: MinMax(H,sym ='Lmb',typ='E',k=k)
    return k,ni,mi,H,Comp(H)

def task5LmbV(key):
    '''
    Task meant to be used as a wrapper to ompute a components value for Lmb_V^k
    also checks if graph is complete and does a shortcut if so
    '''
    k,ni,mi,H = key
    maxn = Choose(ni,2)
    if mi==maxn:
        LmbVkComplete(ni,k)
    elif mi==(maxn-1):
        LmbVkCompleteMinusOne(ni,k)
    #define the computation I want
    Comp = lambda H: MinMax(H,sym ='Lmb',typ='V',k=k)
    return k,ni,mi,H,Comp(H)

#the above task are meant to be passed into multithreading expressions

def GetGG(n_max,Multi=False,threads=2):
    '''
    Generates a dictionary that maps
    Dict[n][m] => Gnm set
    '''
    #initate a GG dict
    GG = {}

    #also make a keys list
    Keys = []

    Nrange = range(1,n_max+1)
    #premake all the keys
    for ni in Nrange:
        GG[ni] = {}
        for mi in range(0, Choose(ni,2)+1 ):
            GG[ni][mi] = None
            Keys.append((ni,mi))

    #now GG is made such that GG[n][m] => None and Keys contains unique (n,m) pairs

    #fill out GG without multithreading
    if not Multi:
        for key in Keys:
            #make the Gnm set, usualy this is a generator
            n,m = key
            Gnm = list(graphs(n,size=m))
            GG[n][m] = Gnm
    #fillout GG using multithreading
    else:
        #task for mapping onto workers is defined in higher scope

        #run on workers
        with concurrent.futures.ProcessPoolExecutor(max_workers = int(threads)) as executor:
            ress = executor.map(task,Keys)
            for res in ress:
                if res==None:
                    pass
                else:
                    GG[res[1][0]][res[1][1]] = res[0]
        

    return GG

def GetGGSingleN(n,Multi=False,threads=2):
    '''
    Computes GG file with same strucutre except onlyconsiders a single n
    '''
    #initalize GG file
    GG = {n:{}}
    #compuite the mrange
    Ms = [[n,m] for m in range(0, Choose(n,2)+1 )]
    
    if not Multi:
        for pair in Ms:
            m = pair[1]
            GG[n][m] = list(graphs(n,size=m))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers = int(threads)) as executor:
            ress = executor.map(task2,Ms)
            for res in ress:
                if res==None:
                    pass
                else:
                    GG[n][res[0]] = res[1]

    return GG

def GenGG(n_max,Multi=False,threads=2,dirs=None):
    '''
    Runs GetGG and automaticly saves it to a pickle file named after its n_max paramater
    Also keeps track of how long it touk to generate the Gnm sets
    '''
    #start timing
    st = time.perf_counter()
    #generate a GG dictionary
    GGData = GetGG(n_max=n_max,Multi=Multi,threads=threads)
    #end timer
    ed = time.perf_counter()
    runtime = ed-st
    #package the runtime and data into a GG dictionary
    GG = {'runtime(sec)':runtime, 'Data':GGData}

    #genertae a file name based on n_max
    if dirs is None:
        filename = f'GG_n{n_max}.pkl'
    else:
        filename = dirs + f'/GG_n{n_max}.pkl'
    print(f'Saving to file: {filename}')

    #pickle the data
    outfile = open(filename,'wb')
    pickle.dump(GG,outfile)
    outfile.close()

def CompLibSingleK(np,sym,typ,k,dirs=None,prints=False,prints2=False):
    '''
    Looks up a GG file (set of premade Gnm sets) and attempts to precompute the specified
    Parameter across all connected components found anywhere in all these Gnm sets
    Organizes them into a dict with structur
    Dict[n][m][H] => Param(H)
    Also accepts a directory if GG files are stored somewhere
    '''
    #Look for a GG_n#.pkl file
    GraphsData = LoadGG(dirs=dirs,prints=prints)

    #get the graphs data
    GG = GraphsData['Data']

    #initalize where we store component files and the specific computation we do
    CompLib = {}
    Comp = lambda G : MinMax(G,sym=sym,typ=typ,k=k)

    #start a timer
    start = time.perf_counter()
    for m in range(0, Choose(np,2)+1 ):
        #Look for the generator
        Gnm = None
        try:
            Gnm = GG[np][m]
        except:
            print(f'Found GG file does not contain the set G({np},{m})\nWill start to generate one now')
            Gnm = list(graphs(np,size=m))
            print(f"Done Generating G({np},{m})")

        for H in Gnm:
            #remove the singletons from this Graph
            SingletonTrim(H)
            #loop over components of this graph
            for Hi in H.connected_components_subgraphs():
                #make copy of Hi
                Hi = Hi.copy(immutable=True)

                #compute the effective edges and vertices
                n_effective = Hi.order()
                m_effective = Hi.size()

                #see if the CompLib has the edge and vertex pairong
                if n_effective not in CompLib:
                    CompLib[n_effective] = {}
                    #if n isnt there m isnt going to be tere and we can just add the component
                    CompLib[n_effective][m_effective] = {}
                    CompLib[n_effective][m_effective][Hi] = Comp(Hi)
                    continue
                elif m_effective not in CompLib[n_effective]:
                    #in this case n is there m might is not there
                    CompLib[n_effective][m_effective] = {}
                    CompLib[n_effective][m_effective][Hi] = Comp(Hi)
                    continue

                #if the cases didnt catch then the path is there and might be isomorphic to something
                ActualyCompute = True
                for Hcomp in CompLib[n_effective][m_effective].keys():
                    if Hi.is_isomorphic(Hcomp):
                        ActualyCompute = False
                if ActualyCompute:
                    #if we didnt hit a case above compute
                    CompLib[n_effective][m_effective][Hi] = Comp(H)
        if prints2:
            print(f'Edge {m} completed')

    end = time.perf_counter()
    #get runtime in minutes
    runtime = (end-start)
    if prints:
        print(f'{(runtime)} Seconds to compute all {sym}_{typ}^{k} for components up to n={np}')
    return CompLib

def GetCompLib(np,sym,typ,ks,dirs=None,prints=False,prints2=False,Multi=False,threads=2):
    '''
    Runs CompLibSIngleK to an array of Ks specified by the arguments
    can use multiple threads for each k
    '''

    #initate a return dictionary
    NMK = {}
    
    #method for no multiuthreading
    if not Multi:
        for k in ks:
            #run CompLib Single K for this K
            NMK[k] = CompLibSingleK(np=np,sym=sym,typ=typ,k=k,dirs=dirs,prints=prints,prints2=prints)
            if prints:
                print(f"K={k} Finished")
        return NMK
    else:
        #make a keys array that wraps all the arguments
        Keys = [(np,sym,typ,k,dirs,prints,prints2) for k in ks]

        with concurrent.futures.ProcessPoolExecutor(max_workers = int(threads)) as executor:
            ress = executor.map(task3,Keys)
            for res in ress:
                if res==None:
                    pass
                else:
                    NMK[res[0]] = res[1]

        return NMK

def GenCompLib(np,sym,typ,ks,dirs=None,prints=False,prints2=False,Multi=False,threads=2):
    '''
    Wraps GetCompLib and automaticly pickles the file, k's used, and runtime
    '''
    
    ks = [int(k) for k in ks]
    
    #start timing
    st = time.perf_counter()
    #run the GetCompLib for data
    CompLibData = GetCompLib(np=np,sym=sym,typ=typ,ks=ks,dirs=dirs,prints=prints,prints2=prints2,Multi=Multi,threads=threads)
    en = time.perf_counter()
    runtime = en-st

    #this snippit of code geenrate a string of ks as list if they arentconsequtive
    #or a mink-maxk string if they are
    lencond = len( set( npp.diff( ks ) ) ) == 1
    onecond = 1 in npp.unique(npp.diff(ks))

    if len(ks)==1:
        Kstr = f'{ks[0]}'
    elif onecond and lencond:
        Kstr = f'{ks[0]}-{ks[-1]}'
    else:
        Kstr = str(ks[0])
        for k in ks[1:]:
            Kstr += ','+str(k)

    #make a file name CompLib_#_Sym_Typ_k###.pkl
    if dirs is None:
        filename = f'CompLib_n{np}_{sym}_{typ}_k{Kstr}.pkl'
    else:
        filename = dirs + f'/CompLib_n{np}_{sym}_{typ}_k{Kstr}.pkl'
    print(f'Saving to file: {filename}')

    #Now I want ot save a dictionary with runtime,Data, and K list
    CompLib = {'runtime(sec)':runtime,'Data':CompLibData, 'K':list(ks)}

    #pickle the file
    outfile = open(filename,'wb')
    pickle.dump(CompLib,outfile)
    outfile.close()

def GetCompLibExp(np,sym='Chi',typ='E',Krange=[2],dirs=None,threads=2):
    '''
    Experimental use of GenCompLib that multithreads it more. Can be slower for small n, but otherwise
    spreads computation better over cores
    '''
    #load up the GG files
    GGFile = LoadGG(dirs=dirs)

    #acces the GGData from the GGFile
    GG = GGFile['Data']

    #now I need to init a dictionary I want to divy up the Gnm components into
    Components = {Krange[0]:{}}
    maxn = np
    maxm = max(GG[maxn].keys())

    #make new list just of graphs
    AllGraphs = []
    #For the pruposes of divying up, I get all graphs ina single list
    for Gnm in GG[maxn].values():
        AllGraphs += Gnm

    #for now I just need 1 K since theyll all be the same from there
    #At the same time Ill make the keys
    for ni in range(2,maxn+1):
        Components[Krange[0]][ni] = {}
        for mi in range(0,Choose(ni,2)+1):
            Components[Krange[0]][ni][mi] = {}

    #Now I run task 4 over the threads
    with concurrent.futures.ProcessPoolExecutor(max_workers = int(threads)) as executor:
        ress = executor.map(task4,AllGraphs)
        for res in ress:
            if res==None:
                pass
            else:
                for ni,mi,H in res:
                    #see if this i already isomorphic to something
                    Site = Components[Krange[0]][ni][mi]
                    iso = GraphSetCompare(H,Site)
                    if iso is None:
                        Components[Krange[0]][ni][mi][H] = None

    #now the components should be divyed up into the Components so I now need to clone
    #for each k key
    for k in Krange[1:]:
        Components[k] = deepcopy(Components[Krange[0]])

    #I dont need Allgraphs,GG, or GGFile anymore
    del AllGraphs
    del GG
    del GGFile


    #Now I need to run computations on each of these components
    #make the keys
    Keys = []
    for k in Krange:
        for ni in range(2,maxn+1):
            for mi in range(0,Choose(ni,2)+1):
                for H in Components[k][ni][mi].keys():
                    Keys.append([k,ni,mi,H])

    #now I run task 5 on all the keys eith a diffrent task based on inputs
    if sym=='Chi':
        if typ=='E':
            with concurrent.futures.ProcessPoolExecutor(max_workers = int(threads)) as executor:
                ress = executor.map(task5ChiE,Keys)
                for res in ress:
                    if res==None:
                        pass
                    else:
                        k,ni,mi,H,val = res
                        Components[k][ni][mi][H] = val
        elif typ=='V':
                with concurrent.futures.ProcessPoolExecutor(max_workers = int(threads)) as executor:
                    ress = executor.map(task5ChiV,Keys)
                    for res in ress:
                        if res==None:
                            pass
                        else:
                            k,ni,mi,H,val = res
                            Components[k][ni][mi][H] = val

    elif sym=='Lmb':
        if typ=='E':
                with concurrent.futures.ProcessPoolExecutor(max_workers = int(threads)) as executor:
                    ress = executor.map(task5LmbE,Keys)
                    for res in ress:
                        if res==None:
                            pass
                        else:
                            k,ni,mi,H,val = res
                            Components[k][ni][mi][H] = val
        elif typ=='V':
                with concurrent.futures.ProcessPoolExecutor(max_workers = int(threads)) as executor:
                    ress = executor.map(task5LmbV,Keys)
                    for res in ress:
                        if res==None:
                            pass
                        else:
                            k,ni,mi,H,val = res
                            Components[k][ni][mi][H] = val

    return Components

def GenCompLibExp(np,sym='Chi',typ='E',Krange=[2],dirs=None,threads=2):
    '''
    Wraps GetCompLib experimental
    '''
    #start a timer
    st = time.perf_counter()
    #actualy make the computation
    KNMlib = GetCompLibExp(np=np,sym=sym,typ=typ,Krange=Krange,dirs=dirs,threads=threads)
    en = time.perf_counter()
    runtime = en -st

    #this snippit of code geenrate a string of ks as list if they arentconsequtive
    #or a mink-maxk string if they are
    lencond = len( set( npp.diff( Krange ) ) ) == 1
    onecond = 1 in npp.unique(npp.diff(Krange))

    if len(Krange)==1:
        Kstr = f'{Krange[0]}'
    elif onecond and lencond:
        Kstr = f'{Krange[0]}-{Krange[-1]}'
    else:
        Kstr = str(Krange[0])
        for k in Krange[1:]:
            Kstr += ','+str(k)

    #make a file name CompLib_#_Sym_Typ_k###.pkl
    if dirs is None:
        filename = f'CompLib_n{np}_{sym}_{typ}_k{Kstr}.pkl'
    else:
        filename = dirs + f'/CompLib_n{np}_{sym}_{typ}_k{Kstr}.pkl'
    print(f'Saving to file: {filename}')

    #Now I want ot save a dictionary with runtime,Data, and K list
    CompLib = {'runtime(sec)':runtime,'Data':KNMlib, 'K':list(Krange)}

    #pickle the file
    outfile = open(filename,'wb')
    pickle.dump(CompLib,outfile)
    outfile.close()

##################################################################################################
#Shortuct functions
#I use formulas for COmlete graphs since they are fairly large
def LmbEkComplete(n,k):
    if k>=n:
        return 0
    else:
        return (Choose(n,2)-((math.floor(n/2))*k))
def LmbVkComplete(n,k):
    if k>=n:
        return 0
    else:
        return (n-k-(k%2))
def ChiEkComplete(n,k):
    return (Choose(n,2) - math.floor(((k-1)*(n**2))/(2*k)))
def ChiVkComplete(n,k):
    if k>=n:
        return 0
    else:
        return (n-k)


##These formulas havent been proved but they seem ok and intuitive
def LmbEkCompleteMinusOne(n,k):
    return max(LmbEkComplete(n,k)-1,0)

def LmbVkCompleteMinusOne(n,k):
    if k!=2:
        return LmbVkComplete(n,k)
    else:
        return max(0,LmbVkComplete(n,k)-1)

def ChiEkCompleteMinusOne(n,k):
    return max(0,ChiEkComplete(n,k)-1)

def ChiVkCompleteMinusOne(n,k):
    return max(0,ChiVkComplete(n,k)-1)


###################################################################################################
#Heavy Functions
            
def IsFailed(G,sym="Chi",k=2):
    '''
    Function for determining if G is in a failstate
    '''
    if sym=="Chi":
        Chrom = lambda G: G.chromatic_number()
    elif sym=="Lmb":
        Chrom = lambda G: G.chromatic_index()
    else:
        print(f"Symbol {sym} Not recognized")
        return None
    #at this point Chrom should be set to the correct paramater
    #now we check if the graph has failed or if its trivial
    Bool = (IsTrivial(G)) or (Chrom(G)<=k)
    return Bool

def RemovalSets(G,typ='E'):
    '''
    Function to return (not neccesarily failure) removal sets for a particular graph
    either can be edgesa or vertices
    '''
    if typ=='E':
        remset = G.edges()
        remsetsize = G.size()
    elif typ=='V':
        remset = G.vertices()
        remsetsize = G.order()
    else:
        print(f"typ value {typ}, not found to be valid")
        return None
    
    #now we should have a way to get either the edge or vertex sets
    #remsetsize is also now the largest subset size fot the removal sets

    #I need to make a generator for every subset of zise 1,...,remsetsize
    Gens = []
    for i in range(0,remsetsize+1):
        #make a genearator
        Gens.append(it.combinations(remset,i))
    #now combine the generators to get one big generator
    return it.chain(*Gens)

def MinMax(G,sym='Chi',typ='E',k=2):
    '''
    Calculates the minimum amount of 'typ'-removals to put G in a failure set
    defined by having a chromtic 'Chi' or 'Lmb' less than k
    '''
    #generate the removal sets
    Candidates = RemovalSets(G,typ=typ)
    #now for each of these sets I just need to iterate through them until i get a failure set
    #because of the nature of 'Candidates' once you one its the smallest
    for rmset in Candidates:
        #get the size of this remset
        removals = len(rmset)
        #calculate the subgraph without the things specified in the removal set
        H = RemSubGraph(G,rmset,typ=typ)
        #now I need to calculate the coresponding chromatic value for this subgraph
        Done = IsFailed(H,sym=sym,k=k)
        #once were done we stop this loop and return the size of the removal set
        if Done:
            return removals
    
def MinMaxComp(G,sym='Chi',typ='E',k=2):
    '''
    Broadcast the MinMax Function across the connected cmponents of a graph G
    and returns the sum. This equivelant to calculating MinMax on a graph with an
    additional speed advantage
    '''
    parts = []
    for H in G.connected_components_subgraphs():
        parts.append(MinMax(H,sym=sym,typ=typ,k=k))
    return sum(parts)

def MinMaxLookUp(H,CompLib):
    '''
    Calculates the minimum amount of 'typ'-removals to put G in a failure set
    by looking up component values based on precompute vals in CompLib
    CompLib should be mapping CompLib[n][m][H] => O(H)
    '''
    #first I need a copy of H
    Hc = H.copy(immutable=False)
    #remove all the singletons from Hc
    SingletonTrim(Hc)
    #Now I need to loop over the connected components of Hc
    RunSum = 0
    for Hcomp in Hc.connected_components_subgraphs():
        #I need the order and size to lookup in complib
        ni,mi = Hcomp.order(),Hcomp.size()
        #look up the potential candidates for this component
        Candidates = CompLib[ni][mi]
        #I need to look to see if Hcomp is isomorphic to something in here
        DetectKey = GraphSetCompare(Hcomp,Candidates)
        #now with the detected key pull up the value for this component
        RunSum += Candidates[DetectKey]
    return RunSum
