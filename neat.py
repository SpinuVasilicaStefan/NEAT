from copy import deepcopy
import random
import math
import numpy as np
numar_inovatie = 0
inovatii_generatii = {}
perturbare = 0.7


def sigmoid_activation(self, w):
    return 1.0 / (1.0 + math.exp(-w * 4.9))

def verificare_inovatie(self, inp = 0, out = 0):
    global inovatii_generatii
    global numar_inovatie
    if((inp, out) in inovatii_generatii.keys()):
        inovatie = inovatii_generatii[(inp, out)]
    else:
        numar_inovatie += 1
        inovatii_generatii[(inp, out)] = numar_inovatie
        inovatie = numar_inovatie
    return inovatie


class gena:
    def __init__(self, num, tp):
        self.nume = num
        self.tip = tp

class conexiune:
    def __init__(self, inp, out, wgh, stat):
        self.input = inp
        self.output = out
        self.weight = wgh
        self.status = stat
        inovatie = verificare_inovatie(inp, out)
        self.innovation_number = inovatie
        
        
class genom:
    def __init__(self, gene, conexiuni, fitness):
        self.gene = deepcopy(gene)
        self.conexiuni =  deepcopy(conexiuni)
        self.fitness = fitness
        self.teste = [([0,0], [0]), ([1,0], [1]), ([0,1], [1]), ([1,1], [0])]
    def add_node(self, gena):
        self.gene += [gena]
    def add_conn(self, conn):
        self.conexiuni[conn.innovation_number] = conn
    def get_conn(self):
        return self.conexiuni
    def dezactivare_conexiune(self, inp, out):
        for cheie in self.conexiuni.keys():
            if self.conexiuni[cheie].input == inp and self.conexiuni[cheie].output == out:
                self.conexiuni[cheie].status = 0
                return
    def activare_conexiune(self, inp, out):
        for cheie in self.conexiuni.keys():
            if self.conexiuni[cheie].input == inp and self.conexiuni[cheie].output == out:
                self.conexiuni[cheie].status = 1
                return
    def mutatie_normala_conexiuni(self):
        for i in self.conexiuni.keys():
            if random.randint(0, 9) / 10 < perturbare:
                self.conexiuni[i].weight += random.randint(-1, 1) / 10
            else:
                self.conexiuni[i].weight = random.uniform(0, 1)
    def mutare_structurala_noduri(self):
        layere = []
        maxim = 0
        poz = 0
        for ge in self.gene:
            if ge.tip != 0 and ge.tip != 1:
                layere += [ge.tip]
            if ge.nume > maxim:
                maxim = ge.nume
        if layere == []:
            poz = 2
        else:
            if random.randint(0, 1) == 0:
                poz = layere[random.randint(0, len(layere) - 1)]
            else:
                poz = layere[len(layere) - 1] + 1
        iteratii = 200
        while iteratii > 0:
            iteratii -= 1
            conex = self.conexiuni[random.choice(list(self.conexiuni.keys()))]
            tip1 = 0
            tip2 = 0
            for i in self.gene:
                if i.nume == conex.input:
                    tip1 = i.tip
                if i.nume == conex.output:
                    tip2 = i.tip
            if poz != tip1 and poz != tip2:
                break
        if iteratii == 0:
            return 
        maxim = maxim + 1
        self.dezactivare_conexiune(conex.input, conex.output) #dezactivez conexiunea
        self.add_node(gena(maxim, poz)) #adaug un nod nou
        self.add_conn(conexiune(conex.input, maxim,1, 1))
        self.add_conn(conexiune(maxim, conex.output, conex.weight, 1))
        
    def mutare_structurala_conexiuni(self):
        if random.randint(0, 9) < 6:
            return
        while 1 == 1:
            x = self.gene[random.randint(0, len(self.gene) - 1)]
            y = self.gene[random.randint(0, len(self.gene) - 1)]
            if x.tip == y.tip:
                continue
            ok = 0
            for i in self.conexiuni.keys():
                if (self.conexiuni[i].input == x.nume and self.conexiuni[i].output == y.nume) or (self.conexiuni[i].input == y.nume and self.conexiuni[i].output == x.nume):
                    ok = 1
                    break
            if ok == 0:
                if random.randint(0, 1) == 0:
                    self.add_conn(conexiune(x.nume, y.nume, random.uniform(0, 1), 1 ))
                else:
                    self.add_conn(conexiune(y.nume, x.nume, random.uniform(0, 1), 1 ))
                break
                
    def determinare_fitness(self):
        self.fitness = len(self.gene)
        return self.fitness
        
    def comparatie(self, genom):
        self.determinare_fitness()
        genom.determinare_fitness()
        if self.fitness > genom.fitness:
            return 1
        else:
            return 0
        
    def exces(self, genom):
        lista2 = []
        for i in self.conexiuni.keys():
            if i not in genom.conexiuni.keys() and i > max(genom.conexiuni.keys()):
                lista2 += [self.conexiuni[i]]
        return lista2
    
    def disjuncte(self, genom):
        lista3 = []
        for i in self.conexiuni.keys():
            if i not in genom.conexiuni.keys() and i < max(genom.conexiuni.keys()):
                lista3 += [self.conexiuni[i]]
        return lista3
    
    def similaritate(self, genom):
        prag = 0.6
        c = 0.4
        w = abs(self.determinare_fitness() - genom.determinare_fitness())
        d = len(self.disjuncte(genom)) + len(genom.disjuncte(self))
        e = len(self.exces(genom)) + len(genom.exces(self))
        n = max(len(self.conexiuni.keys()), len(genom.conexiuni.keys()))
        if (d + e) / n + c * w >= prag:
            return 1
        else:
            return 0
        
    def partitionare_layere(self):
        layere = {}
        noduri = {}
        for i in self.gene:
            if i.tip in layere.keys():
                layere[i.tip] += [i]
            else:
                layere[i.tip] = [i]
        for i in layere.keys():
            if i == 0:
                continue
            noduri[i] = []
            for j in layere[i]:
                for k in self.conexiuni.keys():
                    if self.conexiuni[k].output == j.nume and self.conexiuni[k].status == 1:
                        noduri[i] += [(self.conexiuni[k].input, j.nume, self.conexiuni[k].innovation_number)]
        return (deepcopy(layere), deepcopy(noduri))
    
    def feed_forwoard(self, test):
        layere, conn = self.partitionare_layere()
        noduri = {}
        flow = list(sorted(conn.keys()))
        for i in layere:
            for j in range(0, len(layere[i])):
                noduri[layere[i][j].nume] = 0
        for i in range(0, len(layere[0])):
            noduri[layere[0][i].nume] = sigmoid_activation(self, test[0][i])
        for i in flow:
            if i == 0:
                continue
            for j in conn[i]:
                if j[0] not in noduri.keys():
                     noduri[j[0]] = 0
                noduri[j[1]] += sigmoid_activation(self, noduri[j[0]] * self.conexiuni[j[2]].weight)
        for j in conn[1]:
                if j[0] not in noduri.keys():
                     noduri[j[0]] = 0
                noduri[j[1]] += sigmoid_activation(self, noduri[j[0]] * self.conexiuni[j[2]].weight)
        ultimul_layer = []
        for i in range(0, len(layere[1])):
            ultimul_layer += [noduri[layere[1][i].nume]]
        return deepcopy(abs(np.array(ultimul_layer) - np.array(test[1])) * (-1))
    
    def determinare_fitness(self):
        eroare = self.feed_forwoard(self.teste[0])
        for i in range(1, len(self.teste)):
            eroare = eroare + self.feed_forwoard(self.teste[i])
        self.fitness = np.mean(np.array(eroare))
        return np.mean(np.array(eroare))
            
        
        
        
        
class specie:
    def __init__(self, genom):
        self.genomi = deepcopy([genom])
        self.campion = genom
        self.reprezentant = genom
        self.fitness_specie = 0
    def adaugare_genom(self, genom):
        self.genomi += [genom]
    def alegere_reprezentant(self):
        self.reprezentant = self.genomi[random.randint(0, len(self.genomi) - 1)]
    def alegere_campion(self):
        self.fitness_specie = 0
        genom_maxim = self.genomi[0]
        genom_maxim.determinare_fitness()
        self.fitness_specie += genom_maxim.fitness
        for i in range(1, len(self.genomi)):
            self.genomi[i].determinare_fitness()
            self.fitness_specie += self.genomi[i].fitness
            if self.genomi[i].fitness > genom_maxim.fitness:
                genom_maxim = deepcopy(self.genomi[i])
        self.campion = genom_maxim
    def apartenenta_specie(self, genom):
        if self.reprezentant.similaritate(genom) == 1:
            return 1
        return 0
    def fitness_genom_specie(self, genom):
        return genom.determinare_fitness() / len(self.genomi)
    
    def eliminare_underfit(self, prag):
        eliminare = []
        for genom in self.genomi:
            if self.fitness_genom_specie(genom) < prag:
                eliminare += [genom]
        self.genomi = deepcopy(list(set(self.genomi) - set(eliminare)))
            
    def determinare_parinti(self, contor):
        if len(self.genomi) < 2:
            return []
        descendenti = []
        if len(self.genomi) == 2:
            return [(self.genomi[0], self.genomi[1])] * contor
        while contor > 0:
            parinte1 = self.genomi[random.randint(0, len(self.genomi) - 1)]
            parinte2 = self.genomi[random.randint(0, len(self.genomi) - 1)]
            if parinte1 != parinte2:
                descendenti += [(parinte1, parinte2)]
                contor -= 1
        return deepcopy(descendenti)
    
    def verificare_criteriu(self, criteriu):
        for genom in self.genomi:
            if genom.determinare_fitness() > criteriu:
                return 1
        return 0
    
    def maxim_specie(self):
        maxim = -1000
        for i in self.genomi:
            if i.determinare_fitness() > maxim:
                maxim = i.determinare_fitness()
        return maxim
    
    
        
    
    
class specii:
    def __init__(self, sp):
        self.spec = deepcopy(sp)
        
    def adaugare_specie(self, sp):
        self.spec[max(list(self.spec.keys())) + 1] = deepcopy(sp)
        
    
    def creare_specie_noua(self):
        specie_noua = {}
        for i in self.spec.keys():
            self.spec[i].alegere_campion()
            specie_noua[i] = specie(self.spec[i].campion)
        return deepcopy(specie_noua)
    
    def impartire_specii(self, populatie):
        specie_noua = self.creare_specie_noua()
        for genom in populatie:
            ok = 0
            for i in specie_noua.keys():
                if specie_noua[i].apartenenta_specie(genom) == 1:  
                    specie_noua[i].adaugare_genom(genom)
                    ok = 1
            if ok == 0:
                specie_noua[max(list(specie_noua.keys())) + 1] = specie(genom)
        return deepcopy(specie_noua)
    
    def eliminare_specii_goale(self):
        eliminare = []
        for i in self.spec.keys():
            if len(self.spec[i].genomi) == 0:
                eliminare += [i]
        for i in eliminare:
            del self.spec[i]
            
    def eliminare_genomi_slabi(self, prag):
        for i in self.spec.keys():
            self.spec[i].eliminare_underfit(prag)
    
    def specie_maximala(self):
        maxim = 0 
        for i in self.spec.keys():
            if len(self.spec[i].genomi) > maxim:
                maxim = len(self.spec[i].genomi)
        return maxim
            
    def parinti_specii(self):
        maxim = self.specie_maximala()
        contoare = {}
        for i in self.spec.keys():
            contoare [i] = random.randint(2, maxim - len(self.spec[i].genomi) + 2)
        return deepcopy(contoare)
        
    def parinti(self):
        cont = self.parinti_specii()
        lista1 = []
        for i in self.spec.keys():
            lista1 += self.spec[i].determinare_parinti(deepcopy(cont[i]))
        return deepcopy(lista1)
    
    def ordonare_parinti(self):
        prnt = self.parinti()
        for i in range(0, len(prnt)):
            if prnt[i][0].determinare_fitness() < prnt[i][1].determinare_fitness():
                aux = prnt[i][0]
                prnt[i][0] = prnt[i][1]
                prnt[i][1] = aux
        return prnt
    
    def criteriu_oprire(self, criteriu):
        for i in self.spec.keys():
            if self.spec[i].verificare_criteriu(criteriu) == 1:
                return 1
        return 0
    
    def specie_maximala1(self):
        maxi = -999
        for i in self.spec:
            if self.spec[i].maxim_specie() > maxi:
                maxi = self.spec[i].maxim_specie()
        return maxi
        

        
def crossover(genomi):
    genom1 = genomi[0]
    genom2 = genomi[1]
    copil = genom(genom1.gene, {}, 0)
    for i in genom1.conexiuni:
        if i in genom2.conexiuni:
            if random.randint(0,1) == 0:
                copil.add_conn(genom1.conexiuni[i])
            else:
                copil.add_conn(genom2.conexiuni[i])
        else:
            copil.add_conn(genom1.conexiuni[i])
    return copil
    

    