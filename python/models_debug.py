import pickle
from abc import abstractmethod

from gurobipy import *
import math
import numpy as np


class BaseModel(object):
    """
    Base class for models, to be used as coding pattern skeleton.
    Can be used for a model on a single cluster or on multiple clusters"""

    def __init__(self):
        """Initialization of your model and its hyper-parameters"""
        pass

    @abstractmethod
    def fit(self, X, Y):
        """Fit function to find the parameters according to (X, Y) data.
        (X, Y) formatting must be so that X[i] is preferred to Y[i] for all i.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        # Customize what happens in the fit function
        return

    @abstractmethod
    def predict_utility(self, X):
        """Method to call the decision function of your model

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        """
        # Customize what happens in the predict utility function
        return

    def predict_preference(self, X, Y):
        """Method to predict which pair is preferred between X[i] and Y[i] for all i.
        Returns a preference for each cluster.

        Parameters
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements to compare with Y elements of same index
        Y: np.ndarray
            (n_samples, n_features) list of features of elements to compare with X elements of same index

        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of preferences for each cluster. 1 if X is preferred to Y, 0 otherwise
        """
        X_u = self.predict_utility(X)
        Y_u = self.predict_utility(Y)

        return (X_u - Y_u > 0).astype(int)

    def predict_cluster(self, X, Y):
        """Predict which cluster prefers X over Y THE MOST, meaning that if several cluster prefer X over Y, it will
        be assigned to the cluster showing the highest utility difference). The reversal is True if none of the clusters
        prefer X over Y.
        Compared to predict_preference, it indicates a cluster index.

        Parameters
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements to compare with Y elements of same index
        Y: np.ndarray
            (n_samples, n_features) list of features of elements to compare with X elements of same index

        Returns
        -------
        np.ndarray:
            (n_samples, ) index of cluster with highest preference difference between X and Y.
        """
        X_u = self.predict_utility(X)
        Y_u = self.predict_utility(Y)

        return np.argmax(X_u - Y_u, axis=1)

    def save_model(self, path):
        """Save the model in a pickle file. Don't hesitate to change it in the child class if needed

        Parameters
        ----------
        path: str
            path indicating the file in which the model will be saved
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(clf, path):
        """Load a model saved in a pickle file. Don't hesitate to change it in the child class if needed

        Parameters
        ----------
        path: str
            path indicating the path to the file to load
        """
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model


class RandomExampleModel(BaseModel):
    """Example of a model on two clusters, drawing random coefficients.
    You can use it to understand how to write your own model and the data format that we are waiting for.
    This model does not work well but you should have the same data formatting with TwoClustersMIP.
    """

    def __init__(self):
        self.seed = 444
        self.weights = self.instantiate()

    def instantiate(self):
        """No particular instantiation"""
        return []

    def fit(self, X, Y):
        """fit function, sets random weights for each cluster. Totally independant from X & Y.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        np.random.seed(self.seed)
        indexes = np.random.randint(0, 2, (len(X)))
        num_features = X.shape[1]
        weights_1 = np.random.rand(num_features)
        weights_2 = np.random.rand(num_features)

        weights_1 = weights_1 / np.sum(weights_1)
        weights_2 = weights_2 / np.sum(weights_2)
        self.weights = [weights_1, weights_2]
        return self

    def predict_utility(self, X):
        """Simple utility function from random weights.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        """
        return np.stack([np.dot(X, self.weights[0]), np.dot(X, self.weights[1])], axis=1)


class TwoClustersMIP(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self, n_pieces, n_clusters):
        """Initialization of the MIP Variables

        Parameters
        ----------
        n_pieces: int
            Number of pieces for the utility function of each feature.
        n°clusters: int
            Number of clusters to implement in the MIP.
        """
        self.L = n_pieces
        self.K = n_clusters
        self.seed = 123
        self.model = self.instantiate(n_pieces, n_clusters)

    def instantiate(self, n_pieces, n_clusters):
        """Instantiation of the MIP Variables - To be completed."""
        # To be
        self.L =n_pieces
        self.K = n_clusters
        model = Model("Gurobi_1")
        return model


    def fit(self, X, Y):
        """Estimation of the parameters - To be completed.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        #optimize
        m = self.instantiate(5, 2)
        L = self.L
        K = self.K
        self.N_COMPARAISON = X.shape[0]
        self.N_CRITERIA = X.shape[1]
        # Instanciation du modèle
        PRECISION = 0.001
        N_COMPARAISON = self.N_COMPARAISON
        N_CRITERIA = self.N_CRITERIA
        #Nombre de features
        #N_CRITERIA = len(X[0])

        #Nombre de segment/breakpoints dans toutes les fonctions de score, possible de l'adapter pour chaque feature 
        #L = 5
        #Nombre d'occurence dans le dataset/produits traités 
        #N_COMPARAISON = len(X)


        #Nombre de clusters 
        #K=2

        #Majorant contrainte 
        M=4

        #à définir clairement : 


        criteria = [[[(m.addVar(name=f"u_{j}{l}{k}")) for k in range(K)] for l in range(L+1)]for j in range(N_CRITERIA) ]

        sigma_Xplus = [[(m.addVar(name=f"sigma+{i}{k}")) for k in range(K) ] for i in range(N_COMPARAISON)]
        sigma_Xminus = [[(m.addVar(name=f"sigma-{i}{k}")) for k in range(K)] for i in range(N_COMPARAISON)]
        sigma_Yplus = [[(m.addVar(name=f"sigma+{i}{k}")) for k in range(K)] for i in range(N_COMPARAISON)]
        sigma_Yminus = [[(m.addVar(name=f"sigma-{i}{k}")) for k in range(K)] for i in range(N_COMPARAISON)]

        
        z_binary = [[m.addVar(name=f"z_{i}_{k}") for k in range(K) ] for i in range(N_COMPARAISON)]

        #hyperparamètre pour le modèle
        epsilon = PRECISION
        # maj du modèle
        m.update()
       

        min_int_X = X.min(axis=0)
        min_int_Y = Y.min(axis=0)
        mins = np.min(np.vstack((min_int_X, min_int_Y)), axis = 0)

        max_int_X = X.max(axis=0)
        max_int_Y = Y.max(axis=0)
        maxs = np.max(np.vstack((max_int_X, max_int_Y)), axis = 0)

        def segment(i, j, X):
            x = X[i][j]
            #floor permet de faire l'arrondi 
            #donne un chiffre entre 0 et L (ici 5) qui donne l'appartenance à l'un des 6 segment possible 
            #X[i] correspond à une occurence d'un produit traité 
            #j correspond à l'indice de feature 
            #correspond au commentaire de la question 2
            return math.floor(L * (x - mins[j]) / (maxs[j] - mins[j]))

        #cette fonction précédente permettra a priori de trouver le l de la fonction suivante

        #l correspond au tronçon sur lequel est la valeur 
        #retourne l'abscisse de gauche ? du tronçon l 
        #appelé xik dans question 3
        def x_seg(j, seg):
            return mins[j] + seg * (maxs[j] - mins[j]) / L

        #calcule la valeur si(xij)
        #cad le score partiel pour un feature donné 
        # ?? on obtient le score à partir d'une forme ressemblant au développement limité autour de la valeur de la feature sur le tronçon

        def u(i,j,k,X,eval : bool = False):
            get_val = (lambda v: v.X) if eval else (lambda v: v)
            x = X[i][j]
            #segm pour le numéro de tronçon
            
            segm = segment(i, j, X)
            
            x_segm = x_seg(j, segm)
            
            x_segm_1 = x_seg(j, segm +1)
            


            if x == maxs[j]:
                return get_val(criteria[j][L][k]) 
            
            S=0
            S = get_val(criteria[j][segm][k])
            slope = (x - x_segm)/(x_segm_1 - x_segm)
            width = get_val(criteria[j][segm+1][k]) - get_val(criteria[j][segm][k])
            S =+ slope*width

            return S


        #définissons la somme des fonctions de score pour un produit (de rang i)
        def s(i, X, k, eval : bool = False):
            if not eval : 
                return quicksum(u(i,j,k,X,False) for j in range(N_CRITERIA))
            else : 
                return sum(u(i,j,k,X,False) for j in range(N_CRITERIA))



        # Ajout des contraintes



        # contraintes de préférence des universités
        for k in range(K): 

            for i in range(N_COMPARAISON) :
                               

                m.addConstr((s(i, X, k, False) - sigma_Xplus[i][k] + sigma_Xminus[i][k]) - (s(i, Y, k, False) - sigma_Yplus[i][k] + sigma_Yminus[i][k] + epsilon)<= (M*z_binary[i][k]+ epsilon))
                m.addConstr(M*(1-z_binary[i][k])<= (s(i, X, k, False) - sigma_Xplus[i][k] + sigma_Xminus[i][k]) - (s(i, Y, k, False) - sigma_Yplus[i][k] + sigma_Yminus[i][k] + epsilon))


        # Constraint for sigma_Xminus
        for k in range(K):
            for i in range(N_COMPARAISON):
                m.addConstr(sigma_Xminus[i][k] >= 0)

        # Constraint for sigma_Xplus
        for k in range(K):
            for i in range(N_COMPARAISON):
                m.addConstr(sigma_Xplus[i][k] >= 0)

        # Constraint for sigma_Yminus
        for k in range(K):
            for i in range(N_COMPARAISON):
                m.addConstr(sigma_Yminus[i][k] >= 0)

        # Constraint for sigma_Yplus
        for k in range(K):
            for i in range(N_COMPARAISON):
                m.addConstr(sigma_Yplus[i][k] >= 0)


        # contrainte 3
        for k in range(K) :
            for j in range(N_CRITERIA):
                for l in range(L):
                    m.addConstr(criteria[j][l+1][k] - criteria[j][l][k] >= epsilon)


        # contrainte 4
        for k in range(K):
                m.addConstr(quicksum(criteria[j][L][k] for j in range(N_CRITERIA)) ==1)
            
        #m.addConstr(s(i, X, k, False) == 1)

        # contrainte 5
        for k in range(K):
            for j in range(N_CRITERIA):
                #ajustement arbitraire avec 2222 car u() traite toujours des matrices
                mins_ada = np.vstack([np.array([ 2, 2, 2, 2]), mins])
                
                print(mins_ada)
                #mins correspond aux plus petites valeurs possibles pour chaque critère
                m.addConstr(u(1, j,k, mins_ada, False) == 0)



        # Flatten the lists of lists into a single list
        sigma_Xplus_flat = [var for sublist in sigma_Xplus for var in sublist]
        sigma_Xminus_flat = [var for sublist in sigma_Xminus for var in sublist]
        sigma_Yminus_flat = [var for sublist in sigma_Yminus for var in sublist]
        sigma_Yplus_flat = [var for sublist in sigma_Yplus for var in sublist]

        # Objective function
        m.setObjective(sum(sigma_Xplus_flat) + sum(sigma_Xminus_flat) + sum(sigma_Yminus_flat) + sum(sigma_Yplus_flat), GRB.MINIMIZE)
      
        # Paramétrage (mode mute)
        #m.Params.Presolve = 0
        # Résolution du PL
        m.optimize()

        # To be completed
        return

    def predict_utility(self, X):
        """Return Decision Function of the MIP for X. - To be completed.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        """
        # To be completed
        # Do not forget that this method is called in predict_preference (line 42) and therefor should return well-organized data for it to work.
        def segment(i, j, X):
            x = X[i][j]
            #floor permet de faire l'arrondi 
            #donne un chiffre entre 0 et L (ici 5) qui donne l'appartenance à l'un des 6 segment possible 
            #X[i] correspond à une occurence d'un produit traité 
            #j correspond à l'indice de feature 
            #correspond au commentaire de la question 2
            return math.floor(L * (x - mins[j]) / (maxs[j] - mins[j]))

        #cette fonction précédente permettra a priori de trouver le l de la fonction suivante

        #l correspond au tronçon sur lequel est la valeur 
        #retourne l'abscisse de gauche ? du tronçon l 
        #appelé xik dans question 3
        def x_seg(j, seg):
            return mins[j] + seg * (maxs[j] - mins[j]) / L

        #calcule la valeur si(xij)
        #cad le score partiel pour un feature donné 
        # ?? on obtient le score à partir d'une forme ressemblant au développement limité autour de la valeur de la feature sur le tronçon

        def u(i,j,k,X,eval : bool = False):
            get_val = (lambda v: v.X) if eval else (lambda v: v)
            x = X[i][j]
            #segm pour le numéro de tronçon
            
            segm = segment(i, j, X)
            
            x_segm = x_seg(j, segm)
            
            x_segm_1 = x_seg(j, segm +1)
            


            if x == maxs[j]:
                return get_val(criteria[j][L][k]) 
            
            S=0
            S = get_val(criteria[j][segm][k])
            slope = (x - x_segm)/(x_segm_1 - x_segm)
            width = get_val(criteria[j][segm+1][k]) - get_val(criteria[j][segm][k])
            S =+ slope*width

            return S
            
            
            def s(i, X, k, eval : bool = False):
                if not eval : 
                    return quicksum(u(i,j,k,X,False) for j in range(N_CRITERIA))
                else : 
                    return sum(u(i,j,k,X,False) for j in range(N_CRITERIA))
        return 

class HeuristicModel(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self):
        """Initialization of the Heuristic Model.
        """
        self.seed = 123
        self.models = self.instantiate()

    def instantiate(self):
        """Instantiation of the MIP Variables"""
        # To be completed
        return

    def fit(self, X, Y):
        """Estimation of the parameters - To be completed.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        # To be completed
        return

    def predict_utility(self, X):
        """Return Decision Function of the MIP for X. - To be completed.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        """
        # To be completed
        # Do not forget that this method is called in predict_preference (line 42) and therefor should return well-organized data for it to work.
        return
