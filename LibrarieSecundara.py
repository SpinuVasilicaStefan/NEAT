#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
import copy
import math
import numpy as np
import pyspark
import csv
import time

inn_numbers = 0
inovatii_generatii = {}
teste = []

teste = []
with open('./lung_cancer.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        if row[0] == 'Name':
            continue
        if row[len(row) - 1] == '1':
            teste = teste + [([1] + list(map(lambda x: x, row[3:])),[1])]
        else:
            teste = teste + [([1] + list(map(lambda x: x, row[3:])),[0])]
        

for test in teste:
    for i in range(0, len(test[0])):
        if test[0][i] != "?":
            test[0][i] = float(test[0][i])
        else:
            test[0][i] = 1       

def sigmoid(x):
    lista = np.array([x])
    return 1 / (1 + np.exp(-4.9 * x))

class Gena:
    def __init__(self, nume, tip):
        self.nume = nume
        self.tip = tip
    def __eq__(self, gena1):
        if self.nume == gena1.nume:
            return True
        return False
    def __hash__(self):
        return self.nume
    def __str__(self):
        return str(self.nume) + ":" + str(self.tip)
    def __lt__(self, gena_aux):
        if self.nume < gena_aux.nume:
            return True
        else:
            return False

class Conexiune:
    def __init__(self, inp, out, wgh):
        self.input = inp
        self.output = out
        self.weight = wgh
        self.status = 1
        self.innovation_number = self.determinare_inovatie(inp, out)
    def __str__(self):
        return str(self.input) + "," + str(self.output) + "," + str(self.weight) + "," + str(self.innovation_number) + "," + str(self.status) + ";"
    def __eq__(self, conn):
        if self.innovation_number == conn.innovation_number:
            return True
        else:
            return False
    def __hash__(self):
        return self.innovation_number
    def determinare_inovatie(self, inp, out):
        global inovatii_generatii
        global inn_numbers
        if (inp, out) in inovatii_generatii.values():
            for i in inovatii_generatii:
                if inovatii_generatii[i] == (inp, out):
                    return i
        else:
            inn_numbers += 1
            inovatii_generatii[inn_numbers] = (inp, out)
            return inn_numbers

class Genom:
    def __init__(self, gene, conexiuni):
        self.gene = copy.deepcopy(gene)
        self.conexiuni = copy.deepcopy(conexiuni)
        self.fitness = self.feed_forwoard()
        
    def __str__(self):
        text = "Gene: \n"
        for i in self.gene:
            text = text + str(i) + ";"
        text += "\nConexiuni: \n"
        for i in self.conexiuni:
            text = text + str(i) + ";"
        return text
    
    def __lt__(self, other):
        return self.fitness < other.fitness
    
    def __gt__(self, other):
        return self.fitness > other.fitness
    
    def __eq__(self, other):
        return self.fitness == other.fitness
    
    def eliminare_disabled(self):
        de_sters = [n for n in self.conexiuni if n.status == 0]
        for conn in de_sters:
            self.conexiuni.remove(conn)

    
    def adaugare_conexiune(self, i, j):
        verificare_con = 0
        maxim = 0
        for k in range(0, len(self.conexiuni)):
            if self.conexiuni[k].input == i and self.conexiuni[k].output == j:
                if self.conexiuni[k].status == 0:
                    verificare_con = 1
                    maxim = k
                else:
                    return
        if verificare_con == 1:
            self.conexiuni[maxim].status = 1
        else:
            con = Conexiune(i, j, random.uniform(0, 0.5)*random.choice([-1, 1]))
            self.conexiuni += [con]
        
    def modificare_weighturi(self):
        conexiune = random.choice(self.conexiuni)
        verificare = random.uniform(0, 1)
        if verificare <= 0.9:
            conexiune.weight += 0.1 * random.choice([-1, 1]) 
        else:
            conexiune.weight = random.uniform(0, 0.5)*random.choice([-1, 1])

            
    def adaugare_nod(self):
        nume_genomi = [i.nume for i in self.gene]
        conexiuni_valide = [i for i in self.conexiuni if i.status == 1]
        if len(conexiuni_valide) == 0:
            return
        conexiune = random.choice(conexiuni_valide)
        conexiune.status = 0
        gena_noua = Gena(max(nume_genomi) + 1, 2)
        conexiune_noua1 = Conexiune(conexiune.input, gena_noua, 1)
        conexiune_noua2 = Conexiune(gena_noua ,conexiune.output, conexiune.weight)
        self.gene += [gena_noua]
        self.conexiuni += [conexiune_noua1, conexiune_noua2]
        
    def mutare(self):
        verificare = random.randint(0, 100)
        if verificare < 90:
            self.modificare_weighturi()
        verificare = random.randint(0, 100)
        if verificare < 5:
            conex = [(i.input, i.output) for i in self.conexiuni]
            pereche = self.random_nodes()
            if pereche not in conex and (pereche[1], pereche[0]) not in conex:
                self.adaugare_conexiune(*pereche)      
        verificare = random.randint(0, 100)
        if verificare < 3:
            self.adaugare_nod()
        return self

            
    def random_nodes(self):
        start = random.choice([n for n in self.gene if n.tip != 1])
        stop = [n for n in self.gene if n.tip != 0]
        if len(stop) == 0:
            nume = [n.nume for n in self.gene]
            j = Gena(max(nume) + 1, 2)
            self.adaugare_nod()
        else:
            j = random.choice(stop)
        return (start, j)
    
    
    def feed_forwoard(self):
        suma = 0
        for test in teste:
            rezultat = self.ff(test[0])
            rezultat = np.mean(np.array(test[1])) - rezultat
            rezultat = rezultat * rezultat
            suma += rezultat
        return suma / len(teste)

    
    def ff(self, inputs):
        valori = {}
        for i in self.gene:
            valori[i.nume] = 0
        inp = [n for n in self.gene if n.tip == 0]
        inp.sort()
        for i in range(0, len(inp)):
            valori[inp[i].nume] += inputs[i]
        for conex in self.conexiuni:
            if conex.input.tip == 0:
                valori[conex.output.nume] += valori[conex.input.nume] * conex.weight
            else:
                valori[conex.output.nume] += sigmoid(valori[conex.input.nume]) * conex.weight
                
        out = [n for n in self.gene if n.tip == 1]
        return sigmoid(valori[out[0].nume])
        

    
    def clonare(self):
        clona = Genom(self.gene, self.conexiuni)
        return clona

 
class Specie:
    def __init__(self, genomi):
        self.genomi = copy.deepcopy(genomi)
        self.max_fitness = self.determinare_campion()
        self.max_generatie = 1
        self.generatie_curenta = 1            
            
    def determinare_campion(self):
        maxim = 1000
        for genom in self.genomi:
            fitness = genom.fitness
            if fitness < maxim:
                maxim = fitness
        return maxim
    
    
    def determinare_parinti(self):
        copii = []
        if random.uniform(0, 1) < 0.4 or len(self.genomi) == 1:
            copil = random.choice(self.genomi).clonare()
            copil.mutare()
            copii += [(copil,)]
        else:
            mama = random.choice(self.genomi)
            tata = random.choice(self.genomi)
            if mama.fitness < tata.fitness:
                copii += [(mama, tata)]
            else:
                copii += [(tata, mama)]
        return copii
    
            
    def eliminare_genomi(self, tip):
        self.genomi.sort()
        if tip == 1:
            self.genomi = [self.genomi[0]]
        else:
            self.genomi = self.genomi[:int(math.ceil(len(self.genomi) / 4)) + 1]
        
    def fitness_specie(self):
        suma = 0
        for i in self.genomi:
            suma += i.fitness
        return 1.0 - suma / len(self.genomi) 
    
    def genom_maxim(self):
        maxim = 1000
        genom_maxim = 0
        for genom in self.genomi:
            if genom.fitness < maxim:
                maxim = genom.fitness
                genom_maxim = genom
        return (maxim, genom_maxim)
    
    def incrucisare(self, genom1, genom2):
        lista_gene = []
        for i in genom1.gene:
            if i not in genom2.gene:
                lista_gene += [i]
        lista_gene += genom2.gene
        conexiuni = []
        for i in genom1.conexiuni:
            ok = 0
            for j in genom2.conexiuni:
                if i.input == j.input and i.output == j.output:
                    if random.choice([0, 1]) == 0:
                        conexiuni.append(copy.deepcopy(i))
                    else:
                        conexiuni.append(copy.deepcopy(j))
                    ok = 1
                    break
            if ok == 0:
                conexiuni.append(i)
        de_sters = []
        for i in lista_gene:
            if i.nume not in [n.input.nume for n in conexiuni] + [n.output.nume for n in conexiuni]:
                de_sters += [i]
        for i in de_sters:
            lista_gene.remove(i)
        copil = Genom(lista_gene, conexiuni)
        return copil
    
    def distanta_genomi(self, genom1, genom2):
        numere1 = [n.innovation_number for n in genom1.conexiuni]
        numere2 = [n.innovation_number for n in genom2.conexiuni]
        maxn1 = max(numere1)
        maxn2 = max(numere2)
        prag = 3
        d = len([n for n in numere1 if n not in numere2 and n < maxn2] + [n for n in numere2 if n not in numere1 and n < maxn1])
        e = len([n for n in numere1 if n not in numere2 and n > maxn2] + [n for n in numere2 if n not in numere1 and n > maxn1])
        c = 0.4
        n = max(len(numere1), len(numere2))
        w = abs(genom1.fitness - genom2.fitness)
        if (d + e) / n + c * w < prag:
            return 1
        else:
            return 0
        

        
class Specii:
    def __init__(self, spe):
        self.spec = spe
        
    def actualizare_fitness(self):
        for specie in self.spec:
            if specie.genom_maxim()[0] < specie.max_fitness:
                specie.max_fitness = specie.genom_maxim()[0]
                specie.max_generatie = specie.generatie_curenta
                
    def eliminare_specii(self):
        lista1 = []
        for _ in range(0, 15):
            lista1.append(copy.deepcopy(genom_special))
        self.actualizare_fitness()
                
        de_sters = []
        for i in range(0, len(self.spec)):
            if self.spec[i].generatie_curenta - self.spec[i].max_generatie >= 15:
                de_sters += [i]
        for i in de_sters:
            self.spec[i].genomi = lista1
                
        
    def clasificare_genom(self, genom):
        start_time = time.time()
        if len(self.spec) == 0:
            self.spec += [Specie([genom])]
        else:
            for specie in self.spec:
                if len(specie.genomi) == 0:
                    continue
                if len(specie.genomi) < 15 and self.distanta_genomi(genom, specie.genomi[0]) == 1:
                    specie.genomi += [genom]
                    return
            self.spec += [Specie([genom])]
            
    def distanta_genomi(self, genom1, genom2):
        numere1 = [n.innovation_number for n in genom1.conexiuni]
        numere2 = [n.innovation_number for n in genom2.conexiuni]
        maxn1 = max(numere1)
        maxn2 = max(numere2)
        prag = 3
        d = len([n for n in numere1 if n not in numere2 and n < maxn2] + [n for n in numere2 if n not in numere1 and n < maxn1])
        e = len([n for n in numere1 if n not in numere2 and n > maxn2] + [n for n in numere2 if n not in numere1 and n > maxn1])
        c = 0.4
        n = max(len(numere1), len(numere2))
        w = abs(genom1.fitness - genom2.fitness)
        if (d + e) / n + c * w < prag:
            return 1
        else:
            return 0
        
    
    def eliminare_underfit_specii(self, tip):
        for i in self.spec:
            i.eliminare_genomi(tip)
    
    def fitness_specii(self):
        suma = 0
        for i in self.spec:
            suma += i.fitness_specie()
        return suma
    
    def determinare_populatie(self):
        return sum([len(n.genomi) for n in self.spec])
    
    def specie_maximala(self):
        minim = 1000
        specie_minima = 0
        genom_minim = 0
        for specie in self.spec:
            if len(specie.genomi) == 0:
                continue
            if specie.genom_maxim()[0] < minim:
                minim = specie.genom_maxim()[0]
                specie_minima = specie
                genom_minim = specie.genom_maxim()[1]
        return minim
            
            
    def selectie(self):
        pop_finala = self.determinare_populatie()
        lungime_populatie = [len(n.genomi) for n in self.spec]
        k = -1
        copii = []
        fitness_spc = [x.fitness_specie() for x in self.spec]
        fitness_total = self.fitness_specii()
        self.eliminare_underfit_specii(0)
        suma = 0
        for specie in self.spec:
            k += 1
            ratio = fitness_spc[k] / fitness_total
            copil_specie = math.floor(ratio * pop_finala) - 1
            for j in range(0, int(copil_specie)):
                copii += specie.determinare_parinti()
        self.eliminare_underfit_specii(1)
        return copii
    
    def impartire(self, copii):
        for copil in copii:
            self.clasificare_genom(copil)
        for specie in self.spec:
            specie.generatie_curenta += 1

def marcate_cu_cancer(genom):
    contor = 0
    for test in teste:
        rezultat = genom.ff(test[0])
        rezultat = abs(rezultat)
        if rezultat >= 0.5:
            contor += 1
    return contor

def marcate_cu_cancer_adevarate(genom):
    contor = 0
    for test in teste:
        rezultat = genom.ff(test[0])
        rezultat = abs(rezultat)
        if rezultat >= 0.5 and np.mean(np.array(test[1])) == 1:
            contor += 1
    return contor

def nemarcate_cu_cancer(genom):
    contor = 0
    for test in teste:
        rezultat = genom.ff(test[0])
        rezultat = abs(rezultat)
        if rezultat < 0.5:
            contor += 1
    return contor

def precision(genom):
    return marcate_cu_cancer_adevarate(genom) /  marcate_cu_cancer(genom)

def recall(genom):
    return marcate_cu_cancer_adevarate(genom) /  totale_cu_cancer()

def totale_cu_cancer():
    contor = 0
    contor2 = 0
    for test in teste:
        if np.mean(np.array(test[1])) == 1:
            contor += 1
        if np.mean(np.array(test[1])) == 0:
            contor2 += 1
    return contor

