import numpy as np

alphabet=np.array(['Alpha','Bravo','Charlie','Delta','Echo','Foxtrot','Golf','Hotel','India','Juliett', 'Kilo', 'Lima','Mike','Night', 'Oscar','Papa','Quebec','Romeo','Sierra', 'Tango', 'Uniform', 'Victor', 'Whiskey', 'X-ray', 'Yankee', 'Zulu'])

class Exercise(object):
    def __init__(self,voters=2, candidates=6,topK=3, display=print):
        self.voters=voters
        self.candidates=candidates
        self.topK=topK
        self.A=np.zeros((self.candidates,self.voters))
        self.display=display
        self.generate()
        self.propose()
        

    def generate(self,):
        
        for a in range(self.candidates):
            self.A[a,:]=np.random.randn(self.voters)*1.5+np.random.rand(1)*5+2.5
    
        self.A[self.A<0]=0.05
        self.A[self.A>10]=10
        
    def propose(self):
        self.display('Uma consulta é enviada para um sistema de IR genérico e as seguintes listas são obtidas:\n')
        self.display('Resultado\t'+'\t'.join(['Rank%d'%(k+1) for k in range(self.voters)]))
        for i_a in range(self.A.shape[0]):            
            a=alphabet[i_a]
            self.display(a+'\t\t'+'\t'.join(['%.2f'%self.A[i_a,k] for k in range(self.voters)]))
            
        self.display('\nCalcule os top-%d resultados usando os algoritmos MedRank, Fagin e o threshold de Fagin'%self.topK)        
    
    
    def sort_A(self):
        S=[]       
        R=[]
        for i in range(self.A.shape[1]):
            idxs=np.flipud(np.argsort(self.A[:,i]))
            R.append(alphabet[idxs])
            S.append(self.A[idxs,i])
        return np.array(S).T, np.array(R).T
        
    def solve(self, debug=False):
        S,R=self.sort_A()
        self.display('Primeiro, vamos ordenar os dois rankings\n')
        self.display('\t\t'.join(['Rank%d\tScore%d'%(k+1,k+1) for k in range(self.voters)]))
        #self.display('Rank1\tScore1\t\tRank2\tScore2')
        for i_a in range(self.candidates):
            #self.display('%s\t%.2f\t\t%s\t%.2f'%(R[i_a,0],S[i_a,0],R[i_a,1],S[i_a,1]))
            self.display('\t\t'.join(['%s\t%.2f'%(R[i_a,k],S[i_a,k]) for k in range(self.voters)]))
        self.display('\n\n')
        Res,k=self.solve_medrank(S,R,debug)
        self.display('\n\n')
        self.solve_Fagin(S,R,k,debug)
        self.display('\n\n')
        self.solve_Fagin_th(S,R, debug)
        
        
    
    def solve_medrank(self,S,R, debug=False):
        num={}
        voters=S.shape[1]
        candidates=S.shape[0]
        for a in range(candidates):
            num[alphabet[a]]=0
            
        k=0
        K=0
        Res=[]
        self.display('=== MedRank ===')
        while len(Res)<self.topK:
            if k!=0 and debug:
                self.display('A lista não foi completamente preenchida, é necessária outra iteração\n')    
            if debug:
                self.display('Posição do rank: %d'%(k+1))
            for j in range(voters):
                if debug:
                    self.display('Analisando %s'%R[k,j])
                num[R[k,j]]+=1
    
                if num[R[k,j]]>self.voters/2 and R[k,j] not in Res:
                    if debug:
                        self.display('%s foi encontrado em mais da metade dos ranks: inserindo na lista'%R[k,j])
                    Res.append(R[k,j])
                    if len(Res)==self.topK:
                        break
            
            k=k+1
        if debug:
            self.display(' = ranking final = ')
        for i in range(self.topK):
            self.display(Res[i])
        
        return Res,k

    def solve_Fagin(self,S,R,k, debug=False):
    
        alph=np.unique(R[:k,:])
        self.display('=== Algoritmo de Fagin ===')
        if debug:
            self.display('Com o Medrank, nós sabemos que precisamos considerar as primeiras %d posições com acesso sequencial'%k)
    
        #A_alph=np.array([A[np.where(alphabet==a])[0],:] for a in alph])
        #A_alph=np.array([A[np.where(alphabet==a)[0],:] for a in alph])
        mean_alph=np.mean(self.A,axis=1)
        mean_alph=np.array([mean_alph[np.where(alphabet==a)[0][0]] for a in alph])
        
        if debug:
            self.display('Usando o acesso randômico para retornar os scores')
            self.display('Os resultados são %s'%str(alph))
            self.display('Com o seguinte score médio')
            for i in range(mean_alph.size):
                self.display('%s\t%.2f'%(alph[i],mean_alph[i]))
            self.display('\n\n = ranking final =')
        idxs=np.flipud(np.argsort(mean_alph))
        
        self.display('Rank\tScore')
        for i in range(self.topK):
            self.display('%s\t%.2f'%(alph[idxs[i]],mean_alph[idxs[i]]))
            
            
    def solve_Fagin_th(self,S,R, debug=False):
        th=15
        res=Results(self.topK)
        seen={}
        self.display('=== Algoritmo de threshold de Fagin ===')            
        for k in range(self.A.shape[0]):
            if debug:
                self.display('\n==> Posição do Rank: %d'%(k+1))
            for j in range(self.voters):
                
                a=R[k,j]
                if a in seen:
                    continue
                
                score=np.mean(self.A[np.where(alphabet==a)[0],:])
                seen[a]=score
                if debug:
                    self.display('%s - score %.2f' %(a,score))
                if res.num<self.topK or score>min_score:
                    if debug:                        
                        self.display('Inserindo %s'%a)
                    res.put(a,score)
                    min_score=res.min_score()
                if debug:
                    self.display('\nLista top-K atual')
                    self.display(str(res))
                    self.display()
            th=np.mean(S[k,:])
            if debug:
                self.display('Threshold: %.2f'%th)
            if res.num<self.topK and debug:    
                self.display('Lista não completa')    
            elif min_score<th and debug:
                
                self.display('Score mínimo é menor que o threshold')    
            if res.num>=self.topK and min_score>=th:
                if debug:                
                    self.display('Score mínimo e maior que o threshold e a lista foi preenchidad: pare!\n')
                break        
        
        self.display('Rank\tScore')
        for i in range(self.topK):
            self.display('%s\t%.2f'%(res.elements[i],res.scores[i]))
        

    
class Results(object):
    def __init__(self, max_items=3):
        self.max_items=max_items
        self.elements=[]
        self.scores=[]
        self.num=0
    def put(self, name, score):
        if self.num==0:
            self.elements.append(name)
            self.scores.append(score)
            self.num=1
            return
        new_el=[]
        new_sc=[]
        score_inserted=False
        for i in range(self.num):
            if self.scores[i]<score and not score_inserted:
                new_el.append(name)
                new_sc.append(score)
                score_inserted=True
                
            new_el.append(self.elements[i])
            new_sc.append(self.scores[i])
        if not score_inserted:
            new_el.append(name)
            new_sc.append(score)
                
        if len(new_el)>self.max_items:
            new_el=new_el[:self.max_items]
            new_sc=new_sc[:self.max_items]
        self.elements=new_el
        self.scores=new_sc
        self.num=len(self.elements)
        
    def min_score(self):
        return self.scores[-1]
    def __str__(self):
        return '\n'.join(['%d\t%s\t%.2f'%(k+1, self.elements[k],self.scores[k]) for k in range(self.num) ])

        
if __name__=="__main__":    
    E=Exercise(voters=3, candidates=7, topK=4)
    E.solve(True)
    
    