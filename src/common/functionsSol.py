from __builtin__ import staticmethod
from operator import mul
from time import sleep
import hashlib
import time
import gc

class Functions():

    def __init__(self):
        self.tops = {}  # top[n][k] = the top given by the binomial coefficient
        self.count_zeros = {}  # count_zeros[n][k] is the number of zeros needed for n and k be corr-imm
        self.mirror_indexes = {}  #mirror_indexes[2][1] = [1,3), mirror_indexes[3][1] = [1,4), mirror_indexes[3][2] = [4,7)

    @staticmethod
    def binomial(m, n):
    #Binomial Coefficient.
        return (lambda n,k: int(round(reduce(mul, (float(n-i)/(i+1) for i in xrange(k)), 1))))(m, n)

    def get_top(self, n, k):
    #Gets the binomial coefficient, if it doesn't have it calculated, does the calculation.
        if (n, k) in self.tops:
            return self.tops[(n, k)]
        top = self.binomial(n, k)
        self.tops[(n, k)] = top
        return top

    def get_mirror_indexes(self, n, k):
        if (n, k) in self.mirror_indexes:
            return self.mirror_indexes[(n, k)]
        top = self.get_top(n, k)
        ini = 1
        for i in range(1, k):
            ini += self.get_top(n, i)
        self.mirror_indexes[(n, k)] = [ini, ini+top]
        return self.mirror_indexes[(n, k)]

    def new_class(self, c_0, c_1, n, k):
    #Generates a class of size N(vars) from two classes of size N-1(vars)
        if k > n:
            k = n
        sum = 0
        prev_sum = 0
        vals = []
        weight = c_0[0]+c_1[0]
        vals.append(weight)
        for i in xrange(1, k+1):
            top = self.get_top(n-1, i-1)
            for _ in xrange(sum, sum+top):
                val = c_0[prev_sum] - c_1[prev_sum]
                vals.append(val)
                prev_sum += 1

            sum += top

            top2 = self.get_top(n-1, i)
            for _ in xrange(sum, sum+top2):
                try:
                    val = c_0[prev_sum] + c_1[prev_sum]
                    vals.append(val)
                except:
                    print "BE CAREFUL!"
                prev_sum += 1

            sum += top2
            prev_sum -= top2
        return tuple(vals)

    def is_correlation_immune(self, class_aux, n, k):
        if k > n:
            return False
        if class_aux[0] % 2 != 0:  # If the hamming weight is not a multiple of 2, not useful
            return False
        if n not in self.count_zeros:
            self.count_zeros[n] = {}
        if k in self.count_zeros[n]:
            count_zeros = self.count_zeros[n][k]
        else:
            count_zeros = 0
            for i in xrange(1,k+1):
                count_zeros += self.binomial(n, i)
            self.count_zeros[n][k] = count_zeros
        for i in xrange(1, count_zeros+1):
            if class_aux[i] != 0:
                return False
        return True

    @staticmethod
    def print_results(N, K, start_time, correlation, solution, all=False):
        if all:
            #print "Correlation Immune"
            #for n in range(2, N+1):
                #for k in xrange(1, K+1):
            #    if K > n:
            #        break
            #    print "N = %s, K = %s, W = %s => %s" % (n, K, correlation[n][K].keys()[0], correlation[n][K])
            print "Minimum Weight Solutions"
            #for n in range(2, N+1):
                #for k in xrange(1, K+1):
            if K > N:
                return
            num_functions = 0
            for aux_class in solution[N][K]:
                num_functions += solution[N][K][aux_class]
            print "N = %s, K = %s, W = %s, #Functions = %s => %s" % (N, K, solution[N][K].keys()[0][0], num_functions, solution[N][K] if len(str(solution[N][K])) < 1000 else 'Too large...')
        elapsed_time = time.time() - start_time
        print "EAT: %s seconds" % elapsed_time

    @staticmethod
    def load_correlation(n, k, k_extend, correlation, db, weight):
        if k_extend > n:
            k_extend = n
        if n not in correlation:
            correlation[n] = {}
        if k not in correlation[n]:
            correlation[n][k] = {}
        if k_extend not in correlation[n][k]:
            correlation[n][k][k_extend] = {}
        corrs = db.load(n=n, k=k, k_extend=k_extend, w=weight, type='correlation')
        if corrs:
            if weight is None:
                for c in corrs:
                    w = c[0]
                    if w not in correlation[n][k][k_extend]:
                        correlation[n][k][k_extend][w] = {}
                    correlation[n][k][k_extend][w].update({c: corrs[c]})
            else:
                correlation[n][k][k_extend][weight] = corrs

    @staticmethod
    def load_solution(n, k, solution, db):
        if n not in solution:
            solution[n] = {}
        if k not in solution[n]:
            solution[n][k] = {}
        sols = db.load(n=n, k=k, type='solution')
        if sols:
            for s in sols:
                solution[n][k].update({s: sols[s]})

    @staticmethod
    def load_classes(n, weight, k_extend, classes, db):
        if k_extend > n:
            k_extend = n
        if n not in classes:
            classes[n] = {}
        if k_extend not in classes[n]:
            classes[n][k_extend] = {}
        if weight not in classes[n][k_extend]:
            classes[n][k_extend][weight] = {}
        cls = db.load(n=n, w=weight, k_extend=k_extend, type='classes')
        if cls:
            if weight is None:
                for c in cls:
                    w = c[0]
                    classes[n][k_extend][w].update({c: cls[c]})
            else:
                classes[n][k_extend][weight] = cls

    def load_partial_classes(self, n, k, k_extend, weight, partialclasses, db):
        self.load_k_partial_class(n, k, k_extend, weight, partialclasses, db)

    def load_batch_partial_classes(self, n, k, k_extend, weight, partialclasses, lst_partialclasses, db):
        results = db.getBatchPartials(n, k, k_extend, weight, lst_partialclasses)
        self.load_k_partial_class(n, k, k_extend, weight, partialclasses, db, results)

    @staticmethod
    def load_k_partial_class(n, k, k_extend, weight, partialclasses, db, results=None):
        if n not in partialclasses:
            partialclasses[n] = {}
        if k not in partialclasses[n]:
            partialclasses[n][k] = {}
        if k_extend not in partialclasses[n][k]:
            partialclasses[n][k][k_extend] = {}
        if weight not in partialclasses[n][k][k_extend]:
            partialclasses[n][k][k_extend][weight] = {}
        if results is None:
            pcls = db.load(n=n, k=k, k_extend=k_extend, w=weight, type='partialclass')
            if pcls:
                for partial, hashes in pcls.iteritems():
                    partialclasses[n][k][k_extend][weight][partial] = hashes
        else:
            for current_partial, current_hashes in results.items():
                partialclasses[n][k][k_extend][weight][current_partial] = current_hashes

    def load_data(self, n, k, k_extend, correlation, solution, classes, partialclasses, db, weight=None):
        #correlation
        self.load_correlation(n, k, k_extend, correlation, db, weight)
        #solution
        self.load_solution(n, k, solution, db)
        #classes
        self.load_classes(n, weight, k_extend, classes, db)
        #partialclasses
        self.load_partial_classes(n, k, k_extend, weight, partialclasses, db)


    @staticmethod
    def solution_exists(n, k, solution, db):
        return len(db.load(n=n, k=k, type='solution')) > 0 or (n in solution and k in solution[n])

    def save_by_weight(self, n, w, k_extend, db, classes, correlation, partialclass, delete=True):
        #print 'Store by weight (N=%s,W=%s)' % (n, w)
        self.save_classes(classes, n, k_extend, db, delete, True, w)
        self.save_correlation(correlation, n, None, k_extend, db, True, w)
        self.save_partial_class(partialclass, n, k_extend, db, delete)
        #print 'END store by weight (N=%s,W=%s)' % (n,w)

    @staticmethod
    def save_pedigree(pedigree, db):
        for n in pedigree:
            for k_extend in pedigree[n]:
                for hash in pedigree[n][k_extend]:
                    db.save(n=n, k_extend=k_extend, hash=hash, objects=pedigree[n][k_extend], type='pedigree')
                    #TODO: ver de no iterar sobre el hash y que se encargue el save de hacerlo
        for n in pedigree.keys():
            del pedigree[n]
        gc.collect()

    def save(self, n, k, k_extend, correlation, solution, classes, partialclass, allcalculated, db, delete=True):
        #save data and free the memory used
        if n > 0:
            self.save_classes(classes, n, k_extend, db, delete)
            self.save_correlation(correlation, n, k, k_extend, db)
            self.save_solution(solution, n, k, db)
            self.save_partial_class(partialclass, n, k_extend, db, delete)
            self.save_all_calculated(allcalculated, n, k_extend, db)

    @staticmethod
    def save_classes(classes, n, k_extend, db, delete, by_weight=False, weight=None):
        if classes:
            #print 'Saving Classes...'
            if n in classes:
                if by_weight:
                    db.save(n=n, w=weight, k_extend=k_extend, objects=classes[n][k_extend][weight], type='classes')
                else:
                    for w in classes[n][k_extend].keys():
                        db.save(n=n, w=w, k_extend=k_extend, objects=classes[n][k_extend][w], type='classes')
                        del classes[n][k_extend][w]
                        gc.collect()
                if delete:
                    del classes[n]
                    gc.collect()
            #print 'Saving Classes... Done Ok!'

    @staticmethod
    def save_correlation(correlation, n, k, k_extend, db, by_weight=False, weight=None):
        #print 'Saving Correlation...'
        if n in correlation:
            if by_weight:
                for current_k in correlation[n]:
                    if weight in correlation[n][current_k][k_extend]:
                        db.save(n=n, k=current_k, k_extend=k_extend, w=weight, objects=correlation[n][current_k][k_extend][weight], type='correlation')
            else:
                for w in correlation[n][k][k_extend]:
                    db.save(n=n, k=k, k_extend=k_extend, w=w, objects=correlation[n][k][k_extend][w], type='correlation')
            del correlation[n]
            gc.collect()
        #print 'Saving Correlation... Done Ok!'

    @staticmethod
    def save_solution(solution, n, k, db):
        #print 'Saving Solution...'
        if n in solution and k in solution[n]:
            db.save(n=n, k=k, objects=solution[n][k], type='solution')
        #print 'Saving Solution... Done Ok!'

    @staticmethod
    def save_partial_class(partialclass, n, k_extend, db, delete):
        #print 'Saving PartialClass...'
        if n in partialclass:
            for k_aux in partialclass[n]:
                if k_extend in partialclass[n][k_aux]:
                    for w in partialclass[n][k_aux][k_extend].keys():
                        db.save(n=n, w=w, k=k_aux, k_extend=k_extend, objects=partialclass[n][k_aux][k_extend][w], type='partialclass')
                        del partialclass[n][k_aux][k_extend][w]
                        gc.collect()
            if delete:
                del partialclass[n]
                gc.collect()
        #print 'Saving PartialClass... Done Ok!'

    @staticmethod
    def save_all_calculated(allcalculated, n, k_extend, db):
        if n in allcalculated and k_extend in allcalculated[n]:
            for w in allcalculated[n][k_extend]:
                db.save(n=n, w=w, k_extend=k_extend, objects={'calculated': allcalculated[n][k_extend][w]}, type='allcalculated')
    @staticmethod
    def add_solution(n, class_aux, count, solution, k):
        if n not in solution:
            solution[n] = {}
        if k not in solution[n]:
            solution[n][k] = {}
        if class_aux in solution[n][k]:
            solution[n][k][class_aux] += count
        else:
            solution[n][k][class_aux] = count

    def refresh_classes(self, n, k_extend, new_weight, new_class, count, count_aux, classes):
        try:
            hash_new_class = self.hash(new_class)
            classes[n][k_extend][new_weight][hash_new_class][new_class] += count * count_aux
        except:
            if n not in classes:
                classes[n] = {}
                classes[n][k_extend] = {}
                classes[n][k_extend][new_weight] = {}
            elif k_extend not in classes[n]:
                classes[n][k_extend] = {}
                classes[n][k_extend][new_weight] = {}
            elif new_weight not in classes[n][k_extend]:
                classes[n][k_extend][new_weight] = {}
            classes[n][k_extend][new_weight][hash_new_class] = {new_class: count * count_aux}
        return classes[n][k_extend][new_weight][hash_new_class][new_class]

    def refresh_pedigree(self, n, k_extend, new_class, c_0, c_1, pedigree):
        try:
            hash = self.hash(new_class)
            hash0 = self.hash(c_0)
            hash1 = self.hash(c_1)
            pedigree[n][k_extend][hash].append((hash0, hash1))
        except:
            if n not in pedigree:
                pedigree[n] = {}
            if k_extend not in pedigree[n]:
                pedigree[n][k_extend] = {}
            if hash not in pedigree[n][k_extend]:
                pedigree[n][k_extend][hash] = {}
            pedigree[n][k_extend][hash] = [(hash0, hash1)]

    def refresh_mirror_classes(self, n, k_extend, new_weight, new_class, classes, partialClass):
        try:
            calculate_partial = False
            hash_new_class = self.hash(new_class)
            aux = classes[n][k_extend][new_weight][hash_new_class].keys()[0]
            if aux != new_class:
                print 'Problemmmmm:::: Hash Collision!!!!', new_class, aux, hash_new_class, hash(aux)
                sleep(10)
        except:
            calculate_partial = True
        if calculate_partial:
            for k in xrange(1, k_extend+1):
                if k > n:
                    return
                top = self.get_top(n, k)
                ini = 1
                for i in range(1, k):
                    ini += self.get_top(n, i)
                partial_class = new_class[ini:ini+top]
                if n not in partialClass:
                    partialClass[n] = {}
                if k not in partialClass[n]:
                    partialClass[n][k] = {}
                if k_extend not in partialClass[n][k]:
                    partialClass[n][k][k_extend] = {}
                if new_weight not in partialClass[n][k][k_extend]:
                    partialClass[n][k][k_extend][new_weight] = {}
                try:
                    if hash_new_class not in partialClass[n][k][k_extend][new_weight][partial_class]:
                        partialClass[n][k][k_extend][new_weight][partial_class].append(hash_new_class)
                except:
                    partialClass[n][k][k_extend][new_weight][partial_class] = [hash_new_class]

    def add_correlation(self, n, k, k_extend, weight, new_class, count, correlation):
        try:
            hash_new_class = self.hash(new_class)
            correlation[n][k][k_extend][weight][hash_new_class][new_class] += count
        except:
            if n not in correlation:
                correlation[n] = {}
            if k not in correlation[n]:
                correlation[n][k] = {}
            if k_extend not in correlation[n][k]:
                correlation[n][k][k_extend] = {}
            if weight not in correlation[n][k][k_extend]:
                correlation[n][k][k_extend][weight] = {}
            if hash_new_class not in correlation[n][k][k_extend][weight]:
                correlation[n][k][k_extend][weight][hash_new_class] = {}
            if new_class in correlation[n][k][k_extend][weight][hash_new_class]:
                correlation[n][k][k_extend][weight][hash_new_class][new_class] += count
            else:
                correlation[n][k][k_extend][weight][hash_new_class][new_class] = count

    @staticmethod
    def hash(elem):
        #return hash(elem)
        return hashlib.sha256(str(elem)).hexdigest()

    @staticmethod
    def is_calculated(n, w, db, k_extend):
        return db.is_calculated(n, w, k_extend)

    @staticmethod
    def remove_classes(classes, n):
        try:
            del classes[n]
            gc.collect()
        except:
            pass

    @staticmethod
    def remove_correlation(correlation, n, k=None):
        try:
            if k is None:
                del correlation[n]
            else:
                del correlation[n][k]
            gc.collect()
        except:
            pass

    @staticmethod
    def clean_structures(classes, pedigree, partialClass, correlation, less_or_equal=None):
        #classes
        if less_or_equal:
            n_range = [i for i in range(1, less_or_equal+1)]
        else:
            n_range = classes.keys()
        for nc in n_range:
            try:
                del classes[nc]
            except:
                pass
        #pedigree
        if less_or_equal:
            np_range = [i for i in range(1, less_or_equal+1)]
        else:
            np_range = pedigree.keys()
        for np in np_range:
            try:
                del pedigree[np]
            except:
                pass
        #partialClass
        if less_or_equal:
            npc_range = [i for i in range(1, less_or_equal+1)]
        else:
            npc_range = partialClass.keys()
        for npc in npc_range:
            try:
                del partialClass[npc]
            except:
                pass
        #correlation
        if less_or_equal:
            ncn_range = [i for i in range(1, less_or_equal+1)]
        else:
            ncn_range = correlation.keys()
        for ncn in ncn_range:
            try:
                del correlation[ncn]
            except:
                pass
        gc.collect()

    def refresh_structures(self, n, k_extend_max, new_class, c_0, c_1, count, count_aux, pedigree, classes, partialClass):
        new_weight = new_class[0]
        self.refresh_pedigree(n, k_extend_max, new_class, c_0, c_1, pedigree)
        self.refresh_mirror_classes(n, k_extend_max, new_weight, new_class, classes, partialClass)
        self.refresh_classes(n, k_extend_max, new_weight, new_class, count, count_aux, classes)


    def extend_class(self, hash_class_to_extend, n, k, k_from, db, pedigreeStruct=None, classesStruct=None, partialClassStruct=None):
        results = {}
        k_max_to_extend = k if k <= n else n
        k_from_extend = k_from if k_from <= n else n
        if n in classesStruct and k_max_to_extend in classesStruct[n]:
            for w in classesStruct[n][k_max_to_extend]:
                if hash_class_to_extend in classesStruct[n][k_max_to_extend][w]:
                    return classesStruct[n][k_max_to_extend][w][hash_class_to_extend]
        pedigrees = db.load(hash=hash_class_to_extend, n=n, type='pedigree')
        if n == 2:
            for pedigree in pedigrees:
                c_0, count = db.getClass(pedigree[0])
                c_1, count_aux = db.getClass(pedigree[1])
                new_class = self.new_class(c_0, c_1, 2, 2)
                self.refresh_structures(n, 2, new_class, c_0, c_1, count, count_aux, pedigreeStruct, classesStruct, partialClassStruct)
                try:
                    results[new_class] += count * count_aux
                except:
                    results[new_class] = count * count_aux
            return results
        for pedigree in pedigrees:
            left_elems = self.extend_class(pedigree[0], n-1, k, k_from_extend, db, pedigreeStruct, classesStruct, partialClassStruct)
            right_elems = self.extend_class(pedigree[1], n-1, k, k_from_extend, db, pedigreeStruct, classesStruct, partialClassStruct)
            for left_elem, count in left_elems.items():
                for right_elem, count_aux in right_elems.items():
                    new_class = self.new_class(left_elem, right_elem, n, k_max_to_extend)
                    self.refresh_structures(n, k_max_to_extend, new_class, left_elem, right_elem, count, count_aux, pedigreeStruct, classesStruct, partialClassStruct)
                    try:
                        results[new_class] += count * count_aux
                    except:
                        results[new_class] = count * count_aux
        return results

    def calculate_extended_classes(self, class_to_extend, n_of_the_class, extend_from, extend_to, print_results=True,
                                   pedigreeStruct=None, classesStruct=None, partialClassStruct=None):
        from src.database.databaseSol import DB
        db = DB()
        results = {}
        self.extend_class(self.hash(class_to_extend), n_of_the_class, extend_to, extend_from, db, pedigreeStruct, classesStruct, partialClassStruct)
        for current_hash in classesStruct[n_of_the_class][extend_to][class_to_extend[0]]:
            for current_class, current_count in classesStruct[n_of_the_class][extend_to][class_to_extend[0]][current_hash].items():
                results[current_class] = current_count
        if pedigreeStruct is not None:
            #save correlations classes
            db.save(n=n_of_the_class, k=extend_from, k_extend=extend_to, w=class_to_extend[0],
                    objects=classesStruct[n_of_the_class][extend_to][class_to_extend[0]], type='correlation')
            #end of save correlations

            self.save_classes(classesStruct, n_of_the_class, extend_to, db, False)
            self.save_pedigree(pedigreeStruct, db)
            self.save_partial_class(partialClassStruct, n_of_the_class, extend_to, db, False)
        '''for r_aux in results:
            if self.is_correlation_immune(r_aux, n_of_the_class, extend_from):
                try:
                    result[r_aux] += 1
                except:
                    result[r_aux] = 1'''
        if print_results:
            print '#####'*15
            print 'Original class: %s' % str(class_to_extend)
            print '#####'*15
            print 'Extending the class from k=%s to k=%s' % (extend_from, extend_to)
            print '\n'.join([str(k) + ': ' + str(v) for k, v in results.items()])
            print 'Total: %s classes' % len(results)
            print '#####'*15
        else:
            return results.items()

    @staticmethod
    def isCorrelationCalculated(n, k, k_extend, w, db):
        return db.isCorrelationCalculated(n, k, k_extend, w)

    def getSolution(self, n, k, db):
        return db.getSolution(n, k)



if __name__ == '__main__':
    f = Functions()