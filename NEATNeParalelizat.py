#!/usr/bin/env python
# coding: utf-8

# In[2]:


import random
import copy
import math
import numpy as np
import csv
import time
import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import io
conf = (pyspark.SparkConf().setAppName("my_job_name").set("spark.shuffle.service.enabled", "false").set("spark.dynamicAllocation.enabled", "false"))
sc = pyspark.SparkContext.getOrCreate(conf=conf)
sc.stop()
conf = (pyspark.SparkConf().setAppName("my_job_name").set("spark.shuffle.service.enabled", "false").set("spark.dynamicAllocation.enabled", "false"))
sc = pyspark.SparkContext(conf=conf)
sc.addFile("./LibrarieSecundara.py")
sc.addFile("./NEATParalelizat.py")
sc.addFile("./lung_cancer.csv")
from LibrarieSecundara import Gena, Genom, Conexiune, Specie, Specii, sigmoid, teste

g = open("rezultate.txt", "a")

def scriere_in_fisier(genom_parametru, nume_fisier):
    fisier = open(nume_fisier, "w")
    afisare = str(genom_parametru)
    gen = afisare.split("Conexiuni:")[0].split("Gene:")[1]
    gen = gen[2:]
    fisier.write(gen)
    cnx = afisare.split("Conexiuni:")[1]
    cnx = cnx[2:]
    cnx_valide = list(filter(lambda x: x != '', cnx.split(";")))
    for i in cnx_valide:
        fisier.write(i)
        fisier.write("\n")
    fisier.close()
    
def citire_din_fisier(nume_fisier):
    gene = []
    conexiuni = []
    fisier = open(nume_fisier, "r")
    continut = fisier.read()
    date = continut.split("\n")[:-1]
    dictionar = {}
    for gen in date[0].split(";")[:-1]:
        extragere = gen.split(":")
        gene = gene + [Gena(int(extragere[0]), int(extragere[1]))]
        dictionar[extragere[0]] = Gena(int(extragere[0]), int(extragere[1])) 
    for con in date[1:]:
        extragere = con.split(",")
        inp = dictionar[extragere[0].split(":")[0]]
        out = dictionar[extragere[1].split(":")[0]]
        wgh = float(extragere[2])
        conexiuni = conexiuni + [Conexiune(inp, out, wgh)]
    fisier.close()
    return Genom(gene, conexiuni)


def modificare_date_simplu(tst):
    import LibrarieSecundara
    LibrarieSecundara.teste = tst

def modificare_genom(gen):
    global genom_special
    genom_special = gen
    
def modificare_specii(specii_parametru):
    global spc
    spc = specii_parametru

def campion_populatie():
    global spc
    lista = [x.genom_maxim() for x in spc.spec]
    lista.sort()
    return lista[0][1]

def cel_mai_slab():
    lista = [x.genom_maxim() for x in spc.spec]
    lista.sort()
    return lista[len(lista) - 1][1]

def resetare_teste(tst):
    global genom_special
    global spc
    import LibrarieSecundara
    LibrarieSecundara.teste = copy.deepcopy(tst)
    from LibrarieSecundara import teste
    teste = copy.deepcopy(tst)
    nod_output = Gena(1,1)
    bias = Gena(0, 0)
    conn_bias = Conexiune(bias, nod_output, random.uniform(0, 0.5)*random.choice([-1, 1]))
    maxim = 3
    gene = [bias, nod_output]
    conexiuni = [conn_bias]
    for _ in range(0, len(tst[0][0]) - 1):
        gena_noua = Gena(maxim,0)
        maxim += 1
        conn_noua = Conexiune(gena_noua, nod_output, random.uniform(0, 0.5)*random.choice([-1, 1]))
        conexiuni += [conn_noua]
        gene += [gena_noua]
    genom_special = Genom(gene, conexiuni)

    lista1 = []
    for _ in range(0, 15):
        lista1.append(copy.deepcopy(random.choice([ genom_special])))
    specie1= Specie(copy.deepcopy(lista1))
    sp = []
    for _ in range(0, 10):
        sp = sp + [copy.deepcopy(specie1)]
    spc = Specii(copy.deepcopy(sp))
    
teste_auxiliare = []
with open('./data.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        if row[0] == 'id':
            continue
        if row[1] == 'M':
            teste_auxiliare = teste_auxiliare + [([1] + list(map(lambda x: float(x) * 0.001, row[2:])),[1])]
        else:
            teste_auxiliare = teste_auxiliare + [([1] + list(map(lambda x: float(x) * 0.001, row[2:])),[0])]


    
def creare_genom():
    nod_output = Gena(1,1)
    bias = Gena(0, 0)
    conn_bias = Conexiune(bias, nod_output, random.uniform(0, 0.5)*random.choice([-1, 1]))
    maxim = 3
    gene = [bias, nod_output]
    conexiuni = [conn_bias]
    for _ in range(0, len(teste[0][0]) - 1):
        gena_noua = Gena(maxim,0)
        maxim += 1
        conn_noua = Conexiune(gena_noua, nod_output, random.uniform(0, 0.5)*random.choice([-1, 1]))
        conexiuni += [conn_noua]
        gene += [gena_noua]
    return Genom(gene, conexiuni)

genom_special = creare_genom()


def creare_specii():
    lista1 = []
    for _ in range(0, 15):
        lista1.append(copy.deepcopy(random.choice([ genom_special])))
    specie1= Specie(copy.deepcopy(lista1))
    sp = []
    for _ in range(0, 10):
        sp = sp + [copy.deepcopy(specie1)]
    return Specii(copy.deepcopy(sp))

spc = creare_specii()

def crossover(geno):
    if len(geno) == 1:
        return geno[0]
    genom1, genom2 = geno
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

def eliminare_specii_stagnante(specii_parametru):
    lista1 = []
    for _ in range(0, 15):
        lista1.append(copy.deepcopy(genom_special))
    specii_parametru.actualizare_fitness()          
    de_sters = []
    for i in range(0, len(specii_parametru.spec)):
        if specii_parametru.spec[i].generatie_curenta - specii_parametru.spec[i].max_generatie >= 15 or specii_parametru.spec[i].genom_maxim() == 1000:
            de_sters += [i]
    for i in de_sters:
        specii_parametru.spec[i].genomi = [specii_parametru.spec[i].genomi[0]] + lista1[1:]
        
def feed_forwoard(genom):
        suma = 0
        for test in teste:
            rezultat = ff(genom, test[0])
            rezultat = np.mean(np.array(test[1])) - rezultat
            rezultat = rezultat * rezultat
            suma += rezultat
        genom.fitness = suma / len(teste)
        return 1

    
def ff(genom, inputs):
        valori = {}
        valori1 = {}
        for i in genom.gene:
            valori[i.nume] = 0
            valori1[i.nume] = 0
        inp = [n for n in genom.gene if n.tip == 0]
        inp.sort()
        for i in range(0, len(inp)):
            valori[inp[i].nume] += inputs[i]
        for conex in genom.conexiuni:
            if conex.input.tip == 0:
                valori[conex.output.nume] += valori[conex.input.nume] * conex.weight
            else:
                valori[conex.output.nume] += sigmoid(valori[conex.input.nume]) * conex.weight        
        out = [n for n in genom.gene if n.tip == 1]
        return sigmoid(valori[out[0].nume])

def rulare_neat(epoci_dorite, fitness_dorit, tst, nume):
    if tst != []:
        sc.addFile(nume)
        modificare_date_simplu(tst)
    flag = 0
    if epoci_dorite == 0:
        flag = 1
    precedent = 1000
    for i in range(0, len(spc.spec)):
        for genom in spc.spec[i].genomi:
            genom.fitness = genom.feed_forwoard()
    actual = 1000
    timp_evolutie = 0
    timp_fitness = 0
    epoci = 0
    timp_total = time.time()
    while actual > fitness_dorit:
        if flag == 0 and epoci_dorite == epoci:
            break
        eliminare_specii_stagnante(spc)
        g.write("Epoca numarul: " + str(epoci) + "\n")
        print("Epoca numarul: " + str(epoci))
        print([len(n.genomi) for n in spc.spec])
        start_time = time.time()
        copii = spc.selectie()
        print("--- %s secunde selectie ---" % (time.time() - start_time))
        g.write(str("--- %s secunde selectie ---\n" % (time.time() - start_time)))
        start_time = time.time()
        set_paraleliat = list(map(crossover,copii))
        print("--- %s secunde incrucisare ---" % (time.time() - start_time))
        g.write("--- %s secunde incrucisare ---\n" % (time.time() - start_time))
        timp_evolutie = timp_evolutie + time.time() - start_time
        start_time = time.time()
        spc.impartire(set_paraleliat)
        print("--- %s secunde impartire specii ---" % (time.time() - start_time))
        g.write("--- %s secunde impartire specii ---\n" % (time.time() - start_time))
        actual = min([n.genom_maxim()[0] for n in spc.spec])
        g.write("---fitness-ul total este: " + str(1- actual) + "\n")
        print("--- fitness-ul maximal este: " + str(1- actual) + "% ---")
        start_time = time.time()
        epoci += 1
        for i in range(0, len(spc.spec)):
            for genom in spc.spec[i].genomi:
                genom.mutare()
        print("--- %s secunde mutatie ---" % (time.time() - start_time))
        start_time = time.time()
        lista_genomi = []
        for i in range(0, len(spc.spec)):
            for genom in spc.spec[i].genomi:
                x = genom
                lista_genomi += [x]
        genomi_paralelizati = list(map(feed_forwoard, lista_genomi))
        print("--- %s secunde actualizare fitness ---" % (time.time() - start_time))
        g.write("--- %s secunde actualizare fitness ---\n" % (time.time() - start_time))
        timp_fitness = timp_fitness + time.time() - start_time
    for i in range(0, len(spc.spec)):
        for genom in spc.spec[i].genomi:
            genom.eliminare_disabled()
            genom.fitness = genom.feed_forwoard()
    print("timp fitness = " + str(timp_fitness/epoci))
    g.write("timp fitness = " + str(timp_fitness/epoci))
    print("timp selectie = " + str(timp_evolutie/epoci))
    g.write("timp selectie = " + str(timp_evolutie/epoci))
    g.write("Fitness maxim = " + str(1 - actual) + "\n")
    print("Fitness maxim = " + str(1 - actual)  + "%")
    print("--- %s timp total---" % (time.time() - timp_total))
    g.write(str("--- %s timp total ---\n" % (time.time() - timp_total)))
    g.write("Gata o iteratie \n\n\n\n\n")
    return campion_populatie()


scriere_in_fisier(rulare_neat(0, 0.001, [], "./data.csv"), "genom_scris.txt")


# In[ ]:




