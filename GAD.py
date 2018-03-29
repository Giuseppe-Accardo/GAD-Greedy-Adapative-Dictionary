
# coding: utf-8

# In[1]:


import numpy as np
import math

def GAD(X): #Greedy Adaptive Dictionary
    # 1. Inizializzazione
    l=0
    D=[]
    I=[]
    R = np.copy(X)
    flag=0
    # 2. ---- Ciclo ---- quello con l...
    while l< X.shape[0] and flag<=0: #l=<N    print R
        #print "---- R ----"
        #print R
        # 3. Cerca la colonna di con indice di sparsita min (l1 to l2 norm ratio)
        min_eps = float('inf')
        j_min   =   -1
        for j in range(R.shape[1]): # Colonna
            norma1 = norma2 = 0
            for i in range(R.shape[0]): # Riga
                norma1 += abs(R[i][j])
                norma2 += (R[i][j])**2
            norma2 = math.sqrt(norma2)
            eps = norma1/norma2 #Indice sparsita di quella riga
            if min_eps > eps and j not in I: #Escludiamo 1
                min_eps=eps
                j_min = j
        # 4. Setta l'atomo con la sua normalizzazione
        norma2 = np.sqrt(np.sum(R[:, j_min]**2, axis=0)) #N B FORSE AXIS
        atomo = R[:, j_min]/norma2   
        # 5. Aggiungi Atomo al Dizionario e la pos della riga
        '''La prima volta l atomo stesso, altrimenti aggiunge "Sotto" (vstack)'''
        if len(D) == 0:
            D = np.asarray(atomo)
        else:
            D = np.vstack((D, atomo.T))

        I.append(j_min)
        # 6. Calcola il Nuovo Residuo per tutte le righe i    
        for j in range(X.shape[1]): # Riga
            R[:, j] = R[:, j]-(atomo*( np.inner(atomo.T, R[:, j])))    #PRODOTTO INTERNO = SIMILE SCALARE (Un numero!!!!)
        # 7. Terminazione -> (testa) l<N
        l = l+1
       
        #print norm( (np.dot(np.dot(D.T,(D)), X))-X,2) # alfa per la fine
        #if norm( (np.dot(np.dot(D.T,(D)), X))-X,2)<0.000000001: 
        #    flag=1
    return D,I
        


# In[5]:


import numpy as np

x = np.random.rand(2000,10)
print x.shape
np.set_printoptions(threshold=np.nan)
np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

print(x)


# In[7]:


#Chiamata Function
x=x.T
np.seterr(all='ignore')
[D, I] = GAD(x)
print "-------------- Dizionario-------------------"
#print D.T
print "-------------- Indici-------------------"
#print I
print "--------------D DT-----------------------"
#print D.shape, x.shape
print np.dot(np.dot(D.T,(D)), x).T
print "----------- x.T-----------------------"


# In[195]:


a =  [[2,2,6]]
b = [[3,3,7]]
c = [[4,4,8]]
d = [[5,5,9]]

a = np.asarray(a)
b = np.asarray(b)
c = np.asarray(c)
d = np.asarray(d)
e = np.hstack((a.T,b.T))
e = np.hstack((e,c.T))
e = np.hstack((e,d.T))
e[:,1]


# In[114]:


from scipy.linalg import norm
a =  [ [ 0.75, 0.33333],[0, 0.75]]
print a
norm(a,2)


# In[101]:


a =  np.asarray([1,2,3])
b = np.asarray([0,1,0])
#np.dot(a,b)
c = np.asarray([[1,2,1],[4,5,0],[7,8,1],[10,11,1],[12,13,0]])
#c = c[[0,1]]
# prendi le prima n-1 colonne c[:,0:c.shape[0]-1]
# prendi l'ultima colonna c[:, -1]
final = []
for i_class in range(2): #Da 0 a 1
    print "CLASSE:"+str(i_class)
    mat_class= []
    for row in c:
        #print i
        if row[-1] == i_class:
            #index_class.append(i) #salva indice della riga
            if len(mat_class) == 0:
                mat_class = row
            else:
                mat_class = np.vstack((mat_class, row))
    #print mat_class[ [0,1] ] #0 e 1 sono indici ritornati rispetto sotto matrice
    # smrs( mat_class[:, c.shape[0]-1] ) Non gli devo passare le label
    if len(final) == 0:
        final = mat_class[ [0,1] ]
    else:
        final = np.vstack((final, mat_class[ [0,1] ]))
        
print c[:, 0:c.shape[1]-1]

#mat_class

