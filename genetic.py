import numpy
import random

#GENETIC ALGORITHM OPERATORS

#Mating function
def mating_pool(pop_inputs, objective, num_parents):
    
    objective = numpy.asarray(objective)
    parents = [[None,None,None, None, None, None, None, None, None]]* num_parents
    for parent_num in range(num_parents):
        best_fit_index = numpy.where(objective == numpy.max(objective))
        best_fit_index = best_fit_index[0][0]
        parents[parent_num] = pop_inputs[best_fit_index, :]
        objective[best_fit_index] = -9999999
    return parents

#Crossover function
def crossover(parents, offspring_size):
    
    offspring = [[None,None,None, None, None, None, None, None, None]]* offspring_size[0]
    crossover_loc = numpy.uint32(offspring_size[1]/2)
    parents_list = parents.tolist()
    for k in range(offspring_size[0]):
        # Loc first parent
        parent_1_index = k%parents.shape[0]
        # Loc second parent
        parent_2_index = (k+1)%parents.shape[0]
        # Offspring generation
        offspring[k] = parents_list[parent_1_index][0:crossover_loc] + parents_list[parent_2_index][crossover_loc:]
    return offspring

def mutation(offspring_crossover, sol_per_pop, num_parents_mating, mutation_percent):

    offspring_crossover_a = numpy.asarray(offspring_crossover) # convert to array to do shape calculations
    num_mutations = numpy.uint32((mutation_percent*offspring_crossover_a.shape[1])/100)
    mutation_indices = numpy.array(random.sample(range(0, offspring_crossover_a.shape[1]), num_mutations))
    offspring_mutation = offspring_crossover * sol_per_pop
    offspring_mutation = offspring_mutation [:sol_per_pop-offspring_crossover_a.shape[0]]
    offspring_mutation = numpy.asarray(offspring_mutation, dtype=object)

    for index in range(sol_per_pop-int(num_parents_mating/2)):
        if 0 in mutation_indices: 
            if 1 not in mutation_indices:
                value = random.randint(1,4)
                offspring_mutation[index, 0] = value
                
                if value == 1:  
                    n1 = random.randint(1,20)
                    n = n1
                elif value == 2: 
                    n1 = random.randint(1,20)
                    n2 = random.randint(1,20)
                    n = n1,n2
                elif value == 3: 
                    n1 = random.randint(1,20)
                    n2 = random.randint(1,20)
                    n3 = random.randint(1,20)
                    n = n1,n2,n3
                elif value == 4: 
                    n1 = random.randint(1,20)
                    n2 = random.randint(1,20)    
                    n3 = random.randint(1,20)
                    n4 = random.randint(1,20)
                    n = n1,n2,n3,n4
                offspring_mutation[index, 1] = n
        
        elif [0 and 1] in mutation_indices:
            value = random.randint(1,4)
            offspring_mutation[index, 0] = value
            
            if value == 1:  
                n1 = random.randint(1,20)
                n = n1
            elif value == 2: 
                n1 = random.randint(1,20)
                n2 = random.randint(1,20)
                n = n1,n2
            elif value == 3: 
                n1 = random.randint(1,20)
                n2 = random.randint(1,20)
                n3 = random.randint(1,20)
                n = n1,n2,n3
            elif value == 4: 
                n1 = random.randint(1,20)
                n2 = random.randint(1,20)    
                n3 = random.randint(1,20)
                n4 = random.randint(1,20)
                n = n1,n2,n3,n4
            
            offspring_mutation[index, 1] = n
        
        if 1 in mutation_indices:
            if 0 not in mutation_indices:
                value = random.randint(1,4)
                offspring_mutation[index, 0] = value
                
                if value == 1:  
                    n1 = random.randint(1,20)
                    n = n1
                elif value == 2: 
                    n1 = random.randint(1,20)
                    n2 = random.randint(1,20)
                    n = n1,n2
                elif value == 3: 
                    n1 = random.randint(1,20)
                    n2 = random.randint(1,20)
                    n3 = random.randint(1,20)
                    n = n1,n2,n3
                elif value == 4: 
                    n1 = random.randint(1,20)
                    n2 = random.randint(1,20)    
                    n3 = random.randint(1,20)
                    n4 = random.randint(1,20)
                    n = n1,n2,n3,n4
                
                offspring_mutation[index, 1] = n
            
        if 2 in mutation_indices:
            b = [10, 25, 50, 100, 200]
            value = random.choice(b)
            offspring_mutation[index, 2] = value
            
        if 3 in mutation_indices:
            o = ['Adam', 'Adagrad', 'RMSprop', 'sgd']
            value = random.choice(o)
            offspring_mutation[index, 3] = value

        if 4 in mutation_indices:
            k = ['uniform','normal']
            value = random.choice(k)
            offspring_mutation[index, 4] = value
            
        if 5 in mutation_indices:
            e = [50, 100, 150, 200]
            value = random.choice(e)
            offspring_mutation[index, 5] = value

        if 6 in mutation_indices:
            d = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
            value = random.choice(d)
            offspring_mutation[index, 6] = value
            
        if 7 in mutation_indices:
            t = [0.05, 0.10,0.15,0.20,0.25,0.30]
            value = random.choice(t)
            offspring_mutation[index, 7] = value
        
        if 8 in mutation_indices:
            at = ['relu', 'tanh', 'sigmoid', 'elu']
            value = random.choice(at)
            offspring_mutation[index, 8] = value
     
    return offspring_mutation
