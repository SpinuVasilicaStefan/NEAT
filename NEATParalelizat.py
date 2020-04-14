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

def sigmoid(x):
    x = np.array([x])
    return np.mean(1/(1+ np.exp(-4.9 * x)))

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
        return str(self.nume)
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
        return str(self.input) + ", " + str(self.output) + ", " + str(self.weight) + ", " + str(self.innovation_number) + ", " + str(self.status) + " - "
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
            text = text + str(i) + "; "
        text += "\nConexiuni: \n"
        for i in self.conexiuni:
            text = text + str(i) + "; "
        return text
    
    
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
            con = Conexiune(i, j, random.uniform(0, 1)*random.choice([-1, 1]))
            self.conexiuni += [con]
        
    def adaugare_gena(self, gena):
        self.gene += [gena]
        
    def modificare_weighturi(self):
        conexiune = random.choice(self.conexiuni)
        verificare = random.uniform(0, 1)
        if verificare <= 0.8:
            conexiune.weight += random.uniform(0, 1) * random.choice([-1, 1])
        else:
            conexiune.weight = random.uniform(0, 1)*random.choice([-1, 1])

            
    def adaugare_nod(self):
        nume_genomi = [i.nume for i in self.gene]
        conexiuni_valide = [i for i in self.conexiuni if i.status]
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

            
    def random_nodes(self):
        i = random.choice([n for n in self.gene if n.tip != 1])
        j_list = [n for n in self.gene if n.tip != 0]
        if len(j_list) == 0:
            nume = [n.nume for n in self.gene]
            j = Gena(max(nume) + 1, 2)
            self.adaugare_nod()
        else:
            j = random.choice(j_list)
        return (i, j)
    
    
    def feed_forwoard(self):
        suma = 0
        teste = []
        """with open('C:\\Users\\MrSpV\\Desktop\\data.csv') as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                if row[0] == 'id':
                    continue
                if row[1] == 'M':
                    teste = teste + [([1] + list(map(lambda x: float(x), row[2:])),[1])]
                else:
                    teste = teste + [([1] + list(map(lambda x: float(x), row[2:])),[0])]
        #teste = teste[1:]"""
        teste = [([1,1,1], [0]), ([1,1,0], [1]), ([1,0,1], [1]), ([1,0,0], [0])]
        for test in teste:
            rezultat = self.ff_util(test[0])
            rezultat = np.mean(np.array(test[1])) - rezultat
            rezultat = rezultat * rezultat
            suma += rezultat
        return suma / 4
    
    def ff(self, inputs):
        valori = {}
        valori1 = {}
        for i in self.gene:
            valori[i.nume] = 0
            valori1[i.nume] = 0
        inp = [n for n in self.gene if n.tip == 0]
        inp.sort()
        for i in range(0, len(inp)):
            valori[inp[i].nume] += inputs[i]
        for conex in self.conexiuni:
            valori[conex.output.nume] += sigmoid(valori[conex.input.nume] * conex.weight)

        for conex in self.conexiuni:
            valori1[conex.output.nume] += sigmoid(valori[conex.input.nume] * conex.weight)
            
        #print(valori)
        out = [n for n in self.gene if n.tip == 1]
        return valori1[out[0].nume]
        
    
    def ff_util(self, inputs):
        valori = []
        inp = [n for n in self.gene if n.tip == 0]
        inp.sort()
        coada = [n for n in self.gene if n.tip == 1]
        for i in range(0, len(inp)):
            valori += [(inp[i], inputs[i])]
        ttl = 2000
        while coada != []:
            if ttl == 0:
                return 1000
            ttl -= 1
            nod = coada.pop()
            if nod.tip == 0:
                continue
            adiacente = []
            conexiuni_adiacente = []
            for i in self.conexiuni:
                if i.output.nume == nod.nume and i.status == 1:
                    adiacente += [i.input]
                    conexiuni_adiacente += [i]
            posibile = [n.nume for n in coada]
            for i in adiacente:
                if i.nume not in posibile:
                    coada += adiacente
            ok = 0
            suma = 0
            for i in range(0, len(adiacente)):
                ok1 = 0
                for j in valori:
                    if j[0].nume == adiacente[i].nume:
                        suma =  suma + j[1] * conexiuni_adiacente[i].weight
                        ok1 = 1
                        break
                if ok1 == 0:
                    ok = 1
                    break
            if ok == 1:
                coada = [nod] + coada
            else:
                ok2 = 0
                for i in range(0, len(valori)):
                    if valori[i][0].nume == nod.nume:
                        ok2 = i
                        break
                if ok2 == 0:
                    valori += [(nod, sigmoid(suma))]
                else:
                    valori[i] = (nod, sigmoid(suma))
        return np.mean(np.array([n[1] for n in valori if n[0].tip == 1]))        
    
    def clonare(self):
        clona = Genom(self.gene, self.conexiuni)
        return clona

 
class Specie:
    def __init__(self, genomi):
        self.genomi = copy.deepcopy(genomi)
        self.max_fitness = self.determinare_campion()
        self.max_generatie = self.determinare_campion()
        self.generatie_curenta = 1
        self.moarta = 0
        
    def determinare_superior(self):
        maxim = -100
        for genom in self.genomi:
            fitness = genom.fitness
            if fitness > maxim:
                maxim = fitness
        if maxim > self.max_fitness:
            self.max_fitness = maxim
            self.max_generatie = self.generatie_curenta
            
            
    def determinare_campion(self):
        maxim = -100
        for genom in self.genomi:
            fitness = genom.fitness
            if fitness > maxim:
                maxim = fitness
        return maxim
    
    def determinare_existenta(self):
        if self.generatie_curenta == 1:
            return True
        fitness = self.determinare_campion()
        self.determinare_superior()
        fit = fitness  != self.max_fitness
        end = self.generatie_curenta != self.max_generatie
        return fit and end
    
    def breed(self):
        copilasi = []
        if random.uniform(0, 1) < 0.4 or len(self.genomi) == 1:
            child = random.choice(self.genomi).clonare()
            child.mutare()
            copilasi += [(child,)]
        else:
            mom = random.choice(self.genomi)
            dad = random.choice(self.genomi)
            if mom.fitness < dad.fitness:
                copilasi += [(mom, dad)]
            else:
                copilasi += [(dad, mom)]
        return copilasi
    
            
    def eliminare_genomi(self, tip):
        ok = 1
        ttl = 300
        while ok == 1 and ttl > 0:
            ttl -= 1
            ok = 0
            for i in range(0, len(self.genomi) - 1):
                if self.genomi[i].fitness > self.genomi[i + 1].fitness:
                    self.genomi[i], self.genomi[i + 1] = self.genomi[i + 1], self.genomi[i]
                    ok = 1
        if tip == 1:
            self.genomi = self.genomi[0]
        else:
            self.genomi = self.genomi[:math.ceil(len(self.genomi) / 4)]
        
    def fitness_specie(self):
        suma = 0
        for i in self.genomi:
            suma += i.fitness
        return suma
    
    def genom_maxim(self):
        maxim = 1000
        genom_maxim = 0
        for genom in self.genomi:
            if genom.fitness < maxim:
                maxim = genom.fitness
                genom_maxim = genom
        return (maxim, genom_maxim)
    
    def crossover(self, genom1, genom2):
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
        
    def update_fitness(self):
        for specie in self.spec:
            if specie.genom_maxim()[0] < specie.max_fitness:
                specie.max_fitness = specie.genom_maxim()[0]
                specie.max_generatie = specie.generatie_curenta
                
    def eliminare_specii(self):
        genab = Gena(0, 0)
        gena1 = Gena(1, 0)
        gena2 = Gena(2, 0)
        gena3 = Gena(3, 2)
        gena0 = Gena(5, 2)
        gena4 = Gena(4, 1)
        gena5 = Gena(4, 1)
        obj =  Conexiune(gena1, gena4, random.uniform(0, 1)*random.choice([-1, 1]))
        ob2 =  Conexiune(gena2, gena4, random.uniform(0, 1)*random.choice([-1, 1]))
        ob3 =  Conexiune(genab, gena4, random.uniform(0, 1)*random.choice([-1, 1]))
        gen1 = Genom([genab,gena1, gena2, gena4], [obj, ob2, ob3])

        lista1 = []
        for _ in range(0, 15):
            lista1.append(copy.deepcopy(gen1))
        self.update_fitness()
                
        de_sters = []
        for i in range(0, len(self.spec)):
            if self.spec[i].generatie_curenta - self.spec[i].max_generatie >= 15:
                de_sters += [i]
        for i in de_sters:
            self.spec[i].genomi = lista1
                
        
    def clasificare_genom(self, genom):
        if len(self.spec) == 0:
            self.spec += [Specie([genom])]
        else:
            for specie in self.spec:
                if len(specie.genomi) == 0:
                    continue
                if self.distanta_genomi(specie.genomi[0], genom) == 1 and len(specie.genomi) < 15:
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
            
            
    def evolutie(self):
        pop_finala = self.determinare_populatie()
        lungime_populatie = [len(n.genomi) for n in self.spec]
        k = -1
        self.eliminare_underfit_specii(0)
        copii = []
        for specie in self.spec:
            k += 1
            ratio = specie.fitness_specie() / self.fitness_specii()
            offspring = math.floor(ratio * (pop_finala-lungime_populatie[k]) - 1)
            offspring = 11
            for j in range(0, int(offspring)):
                copii += specie.breed()
        return copii
    
    def impartire(self, copii):
        for copil in copii:
            self.clasificare_genom(copil)
        for specie in self.spec:
            specie.generatie_curenta += 1




