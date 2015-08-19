#! *- encoding: utf-8 -*
from src.common.functionsSol import Functions
from src.database.databaseSol import DB
from time import sleep
from pickle import dumps
import time
import sys


class Problem():

    def __init__(self, n, k):
        if n <= 0 or k <= 0:
            print "N and K must be integers > 0."
            exit_program()
        self.N = n
        self.K = k
        self.k_extend = 1
        self.functions = Functions()
        self.classes = {}
        self.correlation = {}
        self.solution = {}
        self.partialClass = {}
        self.pedigree = {}
        self.allCalculated = {}
        self.start_time = 0
        MB = 1024*1024
        self.limit_of_partial_in_memory = 1024 # xMB
        self.batch_length = 100
        self.db = DB()
        self.init_classes()
        try:
            self.functions.save(1, 1, self.k_extend, self.correlation, self.solution,
                                self.classes, self.partialClass,
                                self.allCalculated, self.db)
            self.functions.save_pedigree(self.pedigree, self.db)
        except:
            print 'The info its already saved'


    def init_classes(self):
        ##############################
        #       THE CLASSES
        ##############################
        #classes[n][k_extend][w][hash] = {class: counter}
        self.classes[1] = {}
        self.classes[1][1] = {}
        self.classes[1][1][0] = {self.functions.hash((0, 0)): {(0, 0): 1}}
        self.classes[1][1][1] = {self.functions.hash((1, -1)): {(1, -1): 1}}
        self.classes[1][1][1].update({self.functions.hash((1, 1)): {(1, 1): 1}})
        self.classes[1][1][2] = {self.functions.hash((2, 0)): {(2, 0): 1}}

        ##############################
        #Information about class generation
        ##############################
        #pedigree[n][k_extend][hash] = [(class_left, class_right)]
        self.pedigree[1] = {}
        self.pedigree[1][1] = {}
        self.pedigree[1][1][self.functions.hash((0, 0))] = [(None, None)]
        self.pedigree[1][1][self.functions.hash((1, -1))] = [(None, None)]
        self.pedigree[1][1][self.functions.hash((1, 1))] = [(None, None)]
        self.pedigree[1][1][self.functions.hash((2, 0))] = [(None, None)]

        ##############################
        #   CORRELATION INMUNE
        ##############################
        #correlation[n][k][k_extend][w][hash] = {class: counter}
        self.correlation[1] = {}
        self.correlation[1][1] = {}
        self.correlation[1][1][1] = {}
        self.correlation[1][1][1][0] = {self.functions.hash((0, 0)): {(0, 0): 1}}
        self.correlation[1][1][1][2] = {self.functions.hash((2, 0)): {(2, 0): 1}}

        ##############################
        #       SOLUTIONS
        ##############################
        #solucion[n][k] = {class: counter}
        self.solution[1] = {}
        self.solution[1][1] = {(2, 0): 1}

        ##############################
        #       PARTIALCLASS
        ##############################
        #partialClass[N][k][k_extend][W][partial_class] = [ids]
        self.partialClass[1] = {}
        self.partialClass[1][1] = {}
        self.partialClass[1][1][1] = {}
        for w in self.classes[1][1]:
            self.partialClass[1][1][1][w] = {}
            for hash in self.classes[1][1][w]:
                for c0, _ in self.classes[1][1][w][hash].iteritems():
                    partial_class = (c0[1],)
                    if partial_class in self.partialClass[1][1][1][w] and self.functions.hash(c0) not in self.partialClass[1][1][1][w][partial_class]:
                        self.partialClass[1][1][1][w][partial_class].append(self.functions.hash(c0))
                    else:
                        self.partialClass[1][1][1][w][partial_class] = [self.functions.hash(c0)]

        ##############################
        # CLASSESS CALCULATED AT ALL
        ##############################
        #allCalculated[N][k_extend][W] = true
        self.allCalculated[1] = {}
        self.allCalculated[1][1] = {}
        self.allCalculated[1][1].update({0: True})
        self.allCalculated[1][1].update({1: True})
        self.allCalculated[1][1].update({2: True})

    def get_all(self, n, weight):
        res = []
        k_extend_max = self.k_extend if self.k_extend <= n else n
        try:
            if n > 0 and (n not in self.classes or weight not in self.classes[n][k_extend_max]):
                self.functions.load_data(n, 1, k_extend_max, self.correlation, self.solution, self.classes,
                                         self.partialClass, self.db, weight)
            if weight <= 2**n and (weight not in self.classes[n][k_extend_max] or not self.functions.is_calculated(n, weight, self.db, k_extend_max)):
                self.get_all_classes(n, weight)
            elif weight not in self.classes[n][k_extend_max]:
                return []
            if n not in self.classes:
                self.functions.load_classes(n, weight, k_extend_max, self.classes, self.db)
            res = [(i, self.classes[n][k_extend_max][weight][hashclass][i]) for hashclass in self.classes[n][k_extend_max][weight]
                        for i in self.classes[n][k_extend_max][weight][hashclass]]
            self.functions.clean_structures(self.classes, self.pedigree, self.partialClass, self.correlation, less_or_equal=n)
        except:
            import traceback
            traceback.print_exc()
        return res

    def get_all_classes(self, n, weight, find_corr=False):
        if n == 0:
            return
        k_extend_max = self.k_extend if self.k_extend <= n else n
        f = self.functions
        # verify if i have the data in memory, if not i will try to obtain the data from db

        if f.is_calculated(n, weight, self.db, k_extend_max) and (n not in self.classes or weight not in self.classes[n][k_extend_max]):
            f.load_data(n, 1, k_extend_max, self.correlation, self.solution, self.classes, self.partialClass, self.db, weight)
            return
        f.remove_classes(self.classes, n)
        # if i do not have the data in memory neither in db, i have to calculate
        if n not in self.classes or weight not in self.classes[n][k_extend_max]:
            print 'Calculating for N = %s and weight = %s' % (n, weight)
            for w in xrange(0, weight+1):
                print "Crossing All Classes of weight %s VS of weight %s --> N = %s" % (w, weight-w, n-1)
                ind = 0
                for c_0, count in self.get_all(n-1, w):
                    for c_1, count_aux in self.get_all(n-1, weight-w):
                        new_class = f.new_class(c_0, c_1, n, k_extend_max)
                        new_weight = new_class[0]
                        f.refresh_pedigree(n, k_extend_max, new_class, c_0, c_1, self.pedigree)
                        f.refresh_mirror_classes(n, k_extend_max, new_weight, new_class, self.classes, self.partialClass)
                        f.refresh_classes(n, k_extend_max, new_weight, new_class, count, count_aux, self.classes)
                        if find_corr and f.is_correlation_immune(new_class, n, 1):
                            f.add_correlation(n, 1, k_extend_max, new_weight, new_class, count * count_aux, self.correlation)
                        ind += 1
            if n not in self.allCalculated:
                self.allCalculated[n] = {}
            if k_extend_max not in self.allCalculated[n]:
                self.allCalculated[n][k_extend_max] = {}
            self.allCalculated[n][k_extend_max].update({weight: True})
            print "END :: Crossing All Classes of weight %s (N=%s)" % (weight, n-1)
            self.functions.save(n, 1, k_extend_max, self.correlation, self.solution, self.classes, self.partialClass, self.allCalculated,
                                self.db, delete=True)  # LEAVE delete IN FALSE!
            self.functions.save_pedigree(self.pedigree, self.db)

    def get_all_k_correlation(self, n, weight, k_corr):
        res = []
        k_extend_max = self.k_extend if self.k_extend <= n else n
        try:
            if k_corr == 0:
                self.get_all_classes(n, weight)
                self.functions.load_classes(n, weight, k_extend_max, self.classes, self.db)
                res = [(i, self.classes[n][k_extend_max][weight][hashclass][i]) for hashclass in self.classes[n][k_extend_max][weight]
                        for i in self.classes[n][k_extend_max][weight][hashclass]]
                return res
            elif k_corr == 1:
                n_iter = n - 1 if n > 1 else 1
                self.get_all_classes(n_iter, weight/2)
                self.generate_k_corr(n, weight, k_corr)
            elif k_corr > 1:
                self.get_all_k_correlation(n-1, weight/2, k_corr-1)
                self.generate_k_corr(n, weight, k_corr)
            self.functions.load_correlation(n, k_corr, k_extend_max, self.correlation, self.db, weight)
            if weight in self.correlation[n][k_corr][k_extend_max]:
                res = [(i, self.correlation[n][k_corr][k_extend_max][weight][hashclass][i]) for hashclass in self.correlation[n][k_corr][k_extend_max][weight]
                        for i in self.correlation[n][k_corr][k_extend_max][weight][hashclass]]
            self.functions.clean_structures(self.classes, self.pedigree, self.partialClass, self.correlation)
        except:
            import traceback
            traceback.print_exc()
        return res

    def generate_k_corr(self, n, weight, k_corr):
        if n==3 and weight==4 and k_corr==2:
            pass
        k_extend_max = self.k_extend if self.k_extend <= n else n
        k_extend_max_load = self.k_extend if self.k_extend <= n - 1 else n - 1
        f = self.functions
        f.clean_structures(self.classes, self.pedigree, self.partialClass, self.correlation)
        if k_corr > 1:
            f.load_correlation(n-1, k_corr-1, k_extend_max_load, self.correlation, self.db, weight/2)
            if weight/2 not in self.correlation[n-1][k_corr-1][k_extend_max_load]:
                return  # There is no k_corr for that N(=n-1) and Weight(=weight)
            set_all_classes = [(i, self.correlation[n-1][k_corr-1][k_extend_max_load][weight/2][hashclass][i]) for hashclass in self.correlation[n-1][k_corr-1][k_extend_max_load][weight/2]
                                for i in self.correlation[n-1][k_corr-1][k_extend_max_load][weight/2][hashclass]]
            self.correlation[n-1][k_corr-1][k_extend_max_load][weight/2] = {}
        else:
            f.load_classes(n-1, weight/2, k_extend_max_load, self.classes, self.db)
            set_all_classes = [(i, self.classes[n-1][k_extend_max_load][weight/2][hashclass][i]) for hashclass in self.classes[n-1][k_extend_max_load][weight/2]
                                for i in self.classes[n-1][k_extend_max_load][weight/2][hashclass]]
            self.classes[n-1][k_extend_max_load][weight/2] = {}
        print 'Generating all %s-corr of weight exactly = %s (N=%s)' % (k_corr, weight, n)
        ind = 0
        ind_added = 0
        size_set = len(set_all_classes)
        for c_0, count in set_all_classes:
            if ind % 100 == 0:
                print '[%s/%s] ---> %s-corr, w=%s, n=%s, Size_Partial=%s' % (ind+1, size_set, k_corr, weight, n, sys.getsizeof(dumps(self.partialClass)))
            if ind % self.batch_length == 0:
                self.mirror_batch(n, k_corr, k_extend_max, set_all_classes, ind, weight)

            for mirror, count_aux in self.find_mirror(c_0, n-1, k_corr):
                new_class = f.new_class(c_0, mirror, n, k_extend_max)
                if f.is_correlation_immune(new_class, n, k_corr):
                    ind_added += 1
                    new_weight = new_class[0]
                    f.refresh_pedigree(n, k_extend_max, new_class, c_0, mirror, self.pedigree)
                    f.refresh_mirror_classes(n, k_extend_max, new_weight, new_class, self.classes, self.partialClass)
                    f.refresh_classes(n, k_extend_max, new_weight, new_class, count, count_aux, self.classes)
                    f.add_correlation(n, k_corr, k_extend_max, new_weight, new_class, count * count_aux, self.correlation)

            ind += 1
        print 'END:: Generating all %s-corr of weight exactly = %s (N=%s)' % (k_corr, weight, n)
        f.save_by_weight(n, weight, k_extend_max, self.db, self.classes, self.correlation, self.partialClass, delete=False)
        f.save_pedigree(self.pedigree, self.db)

    def get_mirror_partial(self, class_aux, n, k):
        top = self.functions.get_top(n, k)
        ini = 1
        for i in range(1, k):
            ini += self.functions.get_top(n, i)
        c0 = class_aux[ini:ini+top]
        partial_class = []
        for elem in c0:
            partial_class.append(elem*-1)
        return tuple(partial_class)

    def find_mirror(self, class_aux, n, k):
        k_extend_max = self.k_extend if self.k_extend <= n else n
        #class = (2,0,0,0,2,alfa) and exists the mirror (2,0,0,0,-2,beta)
        weight = class_aux[0]
        partial_class = self.get_mirror_partial(class_aux, n, k)
        result = []
        if len(partial_class) == 0:
            if k > 1:
                return [(i, self.correlation[n][k-1][k_extend_max][weight][hashclass][i]) for hashclass in self.correlation[n][k-1][k_extend_max][weight]
                            for i in self.correlation[n][k-1][k_extend_max][weight][hashclass]]
            else:
                return [(i, self.classes[n][k_extend_max][weight][hashclass][i]) for hashclass in self.classes[n][k_extend_max][weight]
                                for i in self.classes[n][k_extend_max][weight][hashclass]]
        try:
            for hash_class in self.partialClass[n][k][k_extend_max][weight][partial_class]:
                try:
                    if k > 1:
                        c_aux = self.correlation[n][k-1][k_extend_max][weight][hash_class].keys()[0]
                        result.append((c_aux, self.correlation[n][k-1][k_extend_max][weight][hash_class][c_aux]))
                    else:
                        c_aux = self.classes[n][k_extend_max][weight][hash_class].keys()[0]
                        result.append((c_aux, self.classes[n][k_extend_max][weight][hash_class][c_aux]))
                except:
                    pass
        finally:
            return result

    def calculate_classes(self, n, k_extend, weight):
        self.k_extend = k_extend
        self.get_all_classes(n, weight)
        self.functions.load_classes(n, weight, k_extend, self.classes, self.db)
        size_classes = 0
        size_functions = 0
        for hash in self.classes[n][k_extend][weight]:
            aux_class = self.classes[n][k_extend][weight][hash].keys()[0]
            count = self.classes[n][k_extend][weight][hash][aux_class]
            print '#: %s - class: %s' % (count, aux_class)
            size_classes += 1
            size_functions += count
        print 'Size of classes: %s' % size_classes
        print 'Size of functions: %s\n\n' % size_functions

    def calculate_correlation(self, n, k_extend, k, weight):
        self.k_extend = k_extend
        self.get_all_k_correlation(n, weight, k)
        self.functions.load_correlation(n, k, k_extend, self.correlation, self.db, weight)
        size_correlations = 0
        size_functions = 0
        string_out = ''
        for hash in self.correlation[n][k][k_extend][weight]:
            aux_class = self.correlation[n][k][k_extend][weight][hash].keys()[0]
            count = self.correlation[n][k][k_extend][weight][hash][aux_class]
            current_out = '#: %s - correlation: %s' % (count, aux_class)
            print current_out
            string_out += current_out + '\n'
            size_correlations += 1
            size_functions += count
        file_out = open('out_n%sk%sk_extend%sw%s.txt' % (n, k, k_extend, weight), 'w')
        file_out.write(string_out)
        file_out.write('Size of correlations: %s\n' % size_correlations)
        print 'Size of correlations: %s' % size_correlations
        file_out.write('Size of functions: %s\n\n' % size_functions)
        print 'Size of functions: %s\n\n' % size_functions
        file_out.close()

    def mirror_batch(self, n_iter, k, k_extend, set_classes, ind, weight):
        k_extend_max = k_extend if k_extend <= n_iter-1 else n_iter-1
        #clean partial structure
        if n_iter-1 in self.partialClass:
            del self.partialClass[n_iter-1]
        #clean correlation structure
        if n_iter-1 in self.correlation:
            del self.correlation[n_iter-1]
        #clean classes structure
        if n_iter-1 in self.classes:
            del self.classes[n_iter-1]


        #get batch of the partials
        batch = [self.get_mirror_partial(current_class, n_iter-1, k) for current_class, count in set_classes[ind:ind+self.batch_length]]

        #load all the partials of the batch, in mirror query use the partials structure (dont call to bd!!)
        self.functions.load_batch_partial_classes(n_iter-1, k, k_extend_max, weight/2, self.partialClass, batch, self.db)

        hashes = []
        for elem_i in self.partialClass[n_iter-1][k][k_extend_max][weight/2].values():
            try:
                for hash_i in elem_i:
                    if hash_i not in hashes:
                        hashes.append(hash_i)
            except:
                pass

        if k > 1:
            if len(hashes) > 0:
                #load correlation
                self.db.getBatchCorrelation(n_iter-1, k-1, k_extend_max, hashes, self.correlation)
            else:
                self.functions.load_correlation(n_iter-1, k-1, k_extend_max, self.correlation, self.db, weight/2)

        else:
            if len(hashes) > 0:
                #load classes
                self.db.getBatchClasses(n_iter-1, k_extend_max, hashes, self.classes)
            else:
                self.functions.load_classes(n_iter-1, weight/2, k_extend_max, self.classes, self.db)

        del batch
        del hashes

    def calculate_tree(self, n, k_extend, k, weight, class_tree):
        self.k_extend = k_extend
        self.calculate_classes(n, k_extend, weight)
        self.functions.load_classes(n, weight, k_extend, self.classes,self.db)
        if (class_tree != (-1,)):
            hash = self.functions.hash(class_tree)
            c_aux = self.classes[n][k_extend][weight][hash].keys()[0]
            print 'Class: %s ' % (str(c_aux))
            self.calculate_tree_class(hash, n, k_extend, weight)
        else:
            for hash in self.classes[n][k_extend][weight]:
                #armado de arbol para cada correlations
                c_aux = self.classes[n][k_extend][weight][hash].keys()[0]
                print 'Class: %s ' % (str(c_aux))
                self.calculate_tree_class(hash, n, k_extend, weight)


    def calculate_tree_class(self, hash, n , k_extend, weight):
        k_extend_max = k_extend if k_extend <= (n-1) else (n-1)
        pedigrees = self.db.load(hash=hash, type='pedigree')
        for pedigree in pedigrees:
            left = pedigree[0]
            right = pedigree[1]
            if left is not None and right is not None:
                c_left = self.db.getClass(left)[0]
                c_right = self.db.getClass(right)[0]
                print '%s * %s' % (c_left, c_right)
                print '%s ' %(str(c_left))
                self.calculate_tree_class(left, n-1, k_extend_max, weight/2)
                print '%s ' %(str(c_right))
                self.calculate_tree_class(right, n-1, k_extend_max, weight/2)
            else:
                return



    def solve(self, individual=False):
        self.start_time = time.time()
        f = self.functions
        for n_iter in xrange(2, self.N+1):
            for k in xrange(1, self.N+1):
                if individual:
                    if not (n_iter == self.N and k == self.K):
                        continue
                else:
                    if k > n_iter or (n_iter == self.N and k > self.K):
                        break
                self.k_extend = k
                elapsed_time = time.time() - self.start_time
                #print "EAT: %s seconds" % elapsed_time
                find_solution = False
                weight = self.db.get_last_weight(n_iter, k)
                weight_multiple = 2**k
                persist = False
                if f.solution_exists(n_iter, k, self.solution, self.db):
                    f.load_correlation(n_iter, k, self.k_extend, self.correlation, self.db, weight)
                    f.load_solution(n_iter, k, self.solution, self.db)
                    find_solution = True
                    persist = False
                while not find_solution:
                    try:
                        print ">>>> Weight: %s <-> N = %s, K = %s" % (weight, n_iter, k)
                        f.clean_structures(self.classes, self.pedigree, self.partialClass, self.correlation)

                        if f.isCorrelationCalculated(n_iter-1, k-1, k-1, weight/2, self.db):
                            set_correlations = f.getSolution(n_iter-1, k-1, self.db)
                            if k <= n_iter-1:
                                set_correlations = f.calculate_extended_classes(set_correlations.keys()[0], n_iter-1, k-1, k, False,
                                                                                self.pedigree, self.classes, self.partialClass)
                                self.functions.clean_structures(self.classes, self.pedigree, self.partialClass, self.correlation, True)
                            else:
                                set_correlations = set_correlations.items()
                        else:
                            set_correlations = self.get_all_k_correlation(n_iter-1, weight/2, k-1)


                        ind = 0
                        ind_added = 0
                        print "All %s corr of weight %s VS all of weight %s (N=%s)" % (k-1, weight/2, weight/2, n_iter-1)
                        size_set = len(set_correlations)
                        f.clean_structures(self.classes, self.pedigree, self.partialClass, self.correlation)
                        for c_0, count in set_correlations:
                            if ind % self.batch_length == 0:
                                print '[%s/%s] ---> %s-corr, w=%s, n=%s, Size_Partial=%s' % (ind+1, size_set, k-1, weight/2, n_iter-1, sys.getsizeof(dumps(self.partialClass)))
                            if ind % self.batch_length == 0:
                                self.mirror_batch(n_iter, k, self.k_extend, set_correlations, ind, weight)

                            for mirror, count_aux in self.find_mirror(c_0, n_iter - 1, k):
                                new_class = f.new_class(c_0, mirror, n_iter, self.k_extend)
                                new_weight = new_class[0]
                                if new_weight > 0 and f.is_correlation_immune(new_class, n_iter, k):
                                    ind_added += 1
                                    f.refresh_pedigree(n_iter, self.k_extend, new_class, c_0, mirror, self.pedigree)
                                    f.refresh_mirror_classes(n_iter, self.k_extend, new_weight, new_class, self.classes, self.partialClass)
                                    f.refresh_classes(n_iter, self.k_extend, new_weight, new_class, count, count_aux, self.classes)
                                    f.add_correlation(n_iter, k, self.k_extend, new_weight, new_class, count * count_aux, self.correlation)
                                    f.add_solution(n_iter, new_class, count * count_aux, self.solution, k)
                                    find_solution = True
                                    persist = True

                            ind += 1
                        f.remove_correlation(self.correlation, n_iter-1, k-1)
                        print "All %s corr of weight %s VS all of weight %s - OK (N=%s)" % (k-1, weight/2, weight/2, n_iter-1)
                        weight += weight_multiple
                    except KeyboardInterrupt:
                        exit(1)
                if persist:
                    f.save(n_iter, k, self.k_extend, self.correlation, self.solution, self.classes,
                           self.partialClass, self.allCalculated, self.db, delete=False)
                    f.save_pedigree(self.pedigree, self.db)
                f.print_results(n_iter, k, self.start_time, self.correlation, self.solution, True)


class BooleanFunction():

    def __init__(self, n, k, i):
        self.N = n
        self.K = k
        self.I = i
        self.db = DB()
        self.functions = Functions()
        self.boolean_functions = {self.functions.hash((0, 0)): [0, 0],
                                  self.functions.hash((1, -1)): [0, 1],
                                  self.functions.hash((1, 1)): [1, 0],
                                  self.functions.hash((2, 0)): [1, 1]}

    def solve(self):
        db = self.db
        sols_n_k = db.load(n=self.N, k=self.K, type='solution')
        if self.I is not None:
            sol_n_k = sols_n_k.keys()[0]
            count = sols_n_k[sol_n_k]
            if self.I >= count:
                print "The index is bigger than the maximum allowed..."
                return

            current_count = 0
            for pedigree_sol in db.load(n=self.N, hash=self.functions.hash(sol_n_k), type='pedigree'):
                hash_left = pedigree_sol[0]
                hash_right = pedigree_sol[1]
                if n - 1 == 1:
                    result = self.boolean_functions[hash_left] + self.boolean_functions[hash_right]
                    print ' '.join([str(e) for e in result])
                    return
                class_left, count_left = db.getClass(hash_left)
                class_right, count_right = db.getClass(hash_right)
                current_count += count_left*count_right
                if current_count > self.I:
                    self.I -= current_count - count_left*count_right
                    break
            index_left = self.I / count_right
            index_right = self.I - (index_left * count_right)
            result = self.get_boolean_function(self.N-1, hash_left, index_left) + self.get_boolean_function(self.N-1, hash_right, index_right)
            print ' '.join([str(e) for e in result])
        else:
            for sol_n_k, count in sols_n_k.iteritems():
                result = []
                for pedigree_struct in db.load(n=self.N, hash=self.functions.hash(sol_n_k), type='pedigree'):
                    class_left, count_left = db.getClass(pedigree_struct[0])
                    class_right, count_right = db.getClass(pedigree_struct[1])
                    for i in range(0, count_left):
                        for j in range(0, count_right):
                            lst_left = self.get_boolean_function(self.N-1, pedigree_struct[0], i)
                            lst_right = self.get_boolean_function(self.N-1, pedigree_struct[1], j)
                            result.append(lst_left + lst_right)
                for current_result in result:
                    print ' '.join([str(e) for e in current_result])
                if len(result) > 0:
                    print "Size of result %s" % len(result)

    def get_boolean_function(self, n, hash, index):
        if n == 1:
            return self.boolean_functions[hash]
        current_count = 0
        for pedigree_sol in self.db.load(n=n, hash=hash, type='pedigree'):
            hash_left = pedigree_sol[0]
            hash_right = pedigree_sol[1]
            class_left, count_left = self.db.getClass(hash_left)
            class_right, count_right = self.db.getClass(hash_right)
            current_count += count_left*count_right
            if current_count > index:
                index -= current_count - count_left*count_right
                break
        index_left = index / count_right
        index_right = index - (index_left * count_right)
        return self.get_boolean_function(n-1, hash_left, index_left) + self.get_boolean_function(n-1, hash_right, index_right)

    def get_all_boolean_functions(self, n, hash):
        if n == 1:
            return [self.boolean_functions[hash]]
        result = []
        for pedigree_struct in self.db.load(hash=hash, n=n, type='pedigree'):
            new_hash_left = pedigree_struct[0]
            new_hash_right = pedigree_struct[1]
            lst_left = self.get_all_boolean_functions(n-1, new_hash_left)
            lst_right = self.get_all_boolean_functions(n-1, new_hash_right)
            for left in lst_left:
                for right in lst_right:
                    result.append(left + right)
        return result

if __name__ == '__main__':
    exit_program = False
    while not exit_program:
        try:
            option = input('1\t-> Calculate All Minimum Weights\n' +
                           '11\t-> Calculate Individual Minimum Weight\n' +
                           '12\t-> Calculate Classes\n' +
                           '13\t-> Calculate Correlation\n' +
                           '2\t-> Get Boolean Function\n' +
                           '22\t-> Get All Boolean Functions\n' +
                           '3\t-> Print Solutions\n' +
                           '31\t-> Print Tree of Correlation\n' +
                           '32\t-> Calculate Extended Classes\n' +
                           '400\t-> Drop Database\n' +
                           '5\t-> Exit\n\nSelect option: ')
            if type(option) != int or option < 1:
                continue
        except:  # enter here when option is an string
            continue
        if option == 1:
            try:
                n = input('Select N: ')
                k = input('Select K: ')
            except:
                print 'N and K must be integers > 0'
                continue
            print '(n, k) = (%s,%s)' % (n, k)
            sleep(2)
            Problem(n, k).solve()
        if option == 11:
            try:
                n = input('Select N: ')
                k = input('Select K: ')
            except:
                print 'N and K must be integers > 0'
                continue
            print '(n, k) = (%s,%s)' % (n, k)
            sleep(2)
            Problem(n, k).solve(True)
        if option == 12:
            try:
                n = input('Select N: ')
                k = input('Select K_extend: ')
                w = input('Select W: ')
            except:
                print 'N, K_extend, W must be integers > 0'
                continue
            print '(n, k_extend, w) = (%s,%s,%s)' % (n, k, w)
            sleep(2)
            Problem(n, k).calculate_classes(n, k, w)
        if option == 13:
            try:
                n = input('Select N: ')
                k = input('Select K: ')
                k_extend = input('Select K_extend: ')
                w = input('Select W: ')
            except:
                print 'N, K, K_extend, W must be integers > 0'
                continue
            print '(n, k, k_extend, w) = (%s,%s,%s,%s)' % (n, k, k_extend, w)
            sleep(2)
            Problem(n, k).calculate_correlation(n, k_extend, k, w)
        elif option == 2:
            try:
                n = input('Select N: ')
                k = input('Select K: ')
                i = input('Select I: ')
            except:
                print 'N and K must be integers > 0. I must be an integer >= 0'
                continue
            print '(n, k, i) = (%s,%s,%s)' % (n, k, i)
            sleep(2)
            BooleanFunction(n, k, i).solve()
        elif option == 22:
            try:
                n = input('Select N: ')
                k = input('Select K: ')
            except:
                print 'N and K must be integers > 0.'
                continue
            print '(n, k) = (%s,%s)' % (n, k)
            sleep(2)
            BooleanFunction(n, k, None).solve()

        elif option == 3:
            try:
                n = input('Select N: ')
                k = input('Select K: ')
            except:
                print 'N and K must be integers > 0.'
                continue
            print '(n, k) = (%s,%s)' % (n, k)
            sleep(2)
            functions = Functions()
            solutions = {}
            db = DB()
            functions.load_solution(n,k,solutions,db)
            functions.print_results(n,k,time.time(),None,solutions,True)
        if option == 31:
            try:
                n = input('Select N: ')
                k = input('Select K: ')
                k_extend = input('Select K_extend: ')
                w = input('Select W: ')
                class_tree = raw_input('Class (-1 no class): ')
                class_tree = tuple([int(e) for e in class_tree.split(',')])
            except:
                print 'N, K, K_extend, W must be integers > 0'
                continue
            print '(n, k, k_extend, w) = (%s,%s,%s,%s)' % (n, k, k_extend, w)
            sleep(2)
            Problem(n, k).calculate_tree(n,k_extend,k,w, class_tree)
        if option == 32:
            try:
                class_to_extend = raw_input('Class to extend: ')
                class_to_extend = tuple([int(e) for e in class_to_extend.split(',')])
                n_of_the_class = input('Select N: ')
                extend_from = input('Select original extended k: ')
                extend_to = input('Select K to extend: ')
            except:
                print 'N, K, K_extend, W must be integers > 0'
                continue
            Functions().calculate_extended_classes(class_to_extend, n_of_the_class, extend_from, extend_to, True, {}, {}, {})
        elif option == 5:
            print 'Exit'
            exit_program = True
        elif option == 400:
            print 'dropping database...'
            sleep(2)
            db = DB()
            db.drop_database()
