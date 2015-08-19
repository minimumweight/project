#! *- encoding: utf-8 -*
from pickle import dumps, loads
from pymongo import MongoClient, Connection
from pymongo import ASCENDING as AS
import hashlib


class DB(object):

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DB, cls).__new__(
                                cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self.client = MongoClient('localhost', 27017)
        self.db = self.client.minimumweight
        self.db.classes.ensure_index([('n', AS), ('k_extend', AS), ('w', AS), ('hash', AS)], unique=True)
        self.db.correlation.ensure_index([('n', AS), ('k_extend', AS), ('k', AS), ('w', AS), ('hash', AS)], unique=True)
        self.db.solution.ensure_index([('n', AS), ('k', AS), ('hash', AS)], unique=True)
        self.db.partialclass.ensure_index([('n', AS), ('k', AS), ('k_extend', AS), ('w', AS), ('hash', AS)], unique=True)
        self.db.pedigree.ensure_index([('n', AS), ('k_extend', AS), ('hash', AS), ('data', AS)], unique=True)
        self.db.allcalculated.ensure_index([('n',AS),('w',AS),('k_extend', AS), ('calculated',AS)], unique=True)

    def save(self, **kwargs):
        objects = kwargs.get('objects')
        if len(objects) == 0:
            return
        object_type = kwargs.get('type')
        n = kwargs.get('n')
        w = kwargs.get('w')
        k = kwargs.get('k')
        k_extend = kwargs.get('k_extend')
        hash = kwargs.get('hash')
        ind = 0
        lst = []
        tot = len(objects)

        if object_type == 'correlation':
            db = self.db.correlation
        elif object_type == 'solution':
            db = self.db.solution
        elif object_type == 'classes':
            db = self.db.classes
            current = db.find({'n': n,'w': w, 'k_extend': k_extend}).count()
            if current >= tot:
                return
        elif object_type == 'partialclass':
            db = self.db.partialclass
            current = db.find({'n': n,'w': w, 'k_extend': k_extend, 'k': k}).count()
            if current >= tot:
                return
        elif object_type == 'allcalculated':
            db = self.db.allcalculated
        elif object_type == 'pedigree':
            db = self.db.pedigree
            objects = {hash: objects.get(hash)}
        for key, value in objects.items():
            try:
                if object_type == 'classes':
                    lst.append({"n": n, "w": w, 'k_extend': k_extend, "hash": key, "classes": dumps({key: value})})
                elif object_type == 'correlation':
                    lst.append({"n": n, "k": k, "w": w, 'k_extend': k_extend, "hash": key, "classes": dumps({key: value})})
                elif object_type == 'solution':
                    lst.append({"n": n , "k": k, "hash": self.hash(key), "class": dumps({key: value})})
                elif object_type == 'partialclass':
                    batch = []
                    for elem in value:
                        batch.append(elem)
                        if len(batch) >= 10:
                            lst.append({"n": n, "w": w, 'k': k, 'k_extend': k_extend, "partialclass": dumps(key), "data": dumps(batch), "hash": self.hash(dumps({key:batch}))})
                            batch = []
                    if len(batch) > 0:
                        lst.append({"n": n, "w": w, 'k': k, 'k_extend': k_extend, "partialclass": dumps(key), "data": dumps(batch), "hash": self.hash(dumps({key:batch}))})

                elif object_type == 'allcalculated':
                    lst.append({"n": n, "w": w, 'k_extend': k_extend, "calculated": value})
                elif object_type == 'pedigree':
                    if type(value) == list: #Parche, mejorar desp...
                        for elem in value:
                            lst.append({"n": n, 'k_extend': k_extend, "hash": hash, "data": dumps(elem)})
                    else:
                        lst.append({"n": n, 'k_extend': k_extend, "hash": hash, "data": dumps(value)})
                ind+=1
                if ind%1000==0:
                    print "saving (N=%s)[%s/%s] (type=%s)" % (kwargs.get('n'), ind, tot, object_type)
                    db.insert(lst, {'ordered': False}, continue_on_error=True)
                    lst = []
            except:
                pass
        if len(lst) > 0:
            try:
                db.insert(lst, {'ordered': False}, continue_on_error=True)
            except Exception, e:
                if 'duplicate key error' not in str(e):
                    print 'ERROR: ', str(e)
                else:
                    pass

    def load(self, **kwargs):
        query = {}
        load_items = 'classes'
        type_aux = kwargs.get('type')
        n = kwargs.get('n')
        w = kwargs.get('w')
        k = kwargs.get('k')
        k_extend = kwargs.get('k_extend')
        hash = kwargs.get('hash')
        if n is not None:
            query.update({"n": n})
        if type_aux == 'classes':
            db = self.db.classes
            if w is not None:
                query.update({"w": w})
            if k_extend is not None:
                query.update({'k_extend': k_extend})
        elif type_aux == 'correlation':
            db = self.db.correlation
            if w is not None:
                query.update({"w": w})
            if k is not None:
                query.update({"k": k})
            if k_extend is not None:
                query.update({'k_extend': k_extend})
        elif type_aux == 'solution':
            db = self.db.solution
            if k is not None:
                query.update({"k": k})
            load_items = 'class'
        elif type_aux == 'partialclass':
            db = self.db.partialclass
            load_items = 'partialclass'
            if k_extend is not None:
                query.update({'k_extend': k_extend})
            if w is not None:
                query.update({"w": w})
            if k is not None:
                query.update({'k': k})
        elif type_aux == 'allcalculated':
            db = self.db.allcalculated
            load_items = 'calculated'
            if w is not None:
                query.update({"w": w})
            if k_extend is not None:
                query.update({'k_extend': k_extend})
        elif type_aux == 'pedigree':
            db = self.db.pedigree
            load_items = 'data'
            if k_extend is not None:
                query.update({'k_extend': k_extend})
            if n is not None:
                query.update({"n": n})
            if hash is not None:
                query.update({"hash": hash})


        dic = {}
        skip = 0
        limit = 1000
        tot = db.find(query).count()
        while skip < tot:
            for elem in db.find(query).skip(skip).limit(limit):
                if type_aux == 'partialclass':
                    partial = tuple(loads(str(elem['partialclass'])))
                    hashes = loads(str(elem['data']))
                    if partial not in dic:
                        dic[partial] = hashes
                    elif hashes != dic[partial]:
                        for h in hashes:
                            if h not in dic[partial]:
                                dic[partial].append(h)
                elif type_aux == 'pedigree':
                    if type(dic) == dict:
                        dic = []
                    dic.append(loads(str(elem[load_items])))
                else:
                    for k,v in loads(str(elem[load_items])).items():
                        dic[k] = v
            skip += limit
        return dic

    def get_last_weight(self, n, k):
        db = self.db.solution
        db_solution = db.find_one({"n": n - 1, "k": k-1})
        if db_solution:
            return 2*loads(str(db_solution['class'])).keys()[0][0]
        return 2**k

    def is_calculated(self, n, w, k_extend):
        db = self.db.allcalculated
        return db.find({'n': n, 'w': w, 'k_extend': k_extend}).count() > 0

    def remove_classes(self, n, w):
        db = self.db.classes
        db.remove({'n': n, 'w': w})

    def drop_database(self):
        db = self.db
        c = Connection()
        c.drop_database(db.name)

    def close_mysql(self):
        self.conn.close()

    @staticmethod
    def hash(elem):
        #return hash(elem)
        return hashlib.sha256(str(elem)).hexdigest()

    def getPartials(self, n, k, k_extend, weight, partial_class):
        db = self.db.partialclass
        result = []
        for partial in db.find({'n': n, 'k': k, 'k_extend': k_extend, 'w': weight, 'partialclass': dumps(partial_class)}):
            elements = loads(str(partial['data']))
            for elem in elements:
                if elem not in result:
                    result.append(elem)
        return result

    def getBatchPartials(self, n, k, k_extend, weight, partial_classes):
        db = self.db.partialclass
        lst_partials = [dumps(partial_class) for partial_class in partial_classes]
        results = {}
        for partial in db.find({'n': n, 'k': k, 'k_extend': k_extend, 'w': weight, 'partialclass': {'$in': lst_partials}}):
            elements = loads(str(partial['data']))
            current_partial = loads(str(partial['partialclass']))
            for elem in elements:
                try:
                    if elem not in results[current_partial]:
                        results[current_partial].append(elem)
                except:
                    results[current_partial] = []
                    results[current_partial].append(elem)
        return results

    def getBatchCorrelation(self, n, k, k_extend, hashes, correlation):
        db = self.db.correlation
        for corr in db.find({'n': n, 'k': k, 'k_extend': k_extend, 'hash': {'$in': hashes}}):
            corr_hash = corr['hash']
            weight = corr['w']
            corr_class = loads(str(corr['classes']))[corr_hash]
            if n not in correlation:
                correlation[n] = {}
            if k not in correlation[n]:
                correlation[n][k] = {}
            if k_extend not in correlation[n][k]:
                correlation[n][k][k_extend] = {}
            if weight not in correlation[n][k][k_extend]:
                correlation[n][k][k_extend][weight] = {}
            if corr_hash not in correlation[n][k][k_extend][weight]:
                correlation[n][k][k_extend][weight][corr_hash] = {}
            correlation[n][k][k_extend][weight][corr_hash] = corr_class

    def getBatchClasses(self, n, k_extend, hashes, classes):
        db = self.db.classes
        for class_n in db.find({'n': n, 'k_extend': k_extend, 'hash': {'$in': hashes}}):
            class_hash = class_n['hash']
            db_class = loads(str(class_n['classes']))[class_hash]
            weight = class_n['w']
            if n not in classes:
                classes[n] = {}
            if k_extend not in classes[n]:
                classes[n][k_extend] = {}
            if weight not in classes[n][k_extend]:
                classes[n][k_extend][weight] = {}
            if class_hash not in classes[n][k_extend][weight]:
                classes[n][k_extend][weight][class_hash] = {}
            classes[n][k_extend][weight][class_hash] = db_class

    def getClass(self, hash):
        db = self.db.classes
        class_bd = db.find_one({'hash': hash})
        return loads(str(class_bd['classes']))[hash].items()[0]

    def isCorrelationCalculated(self, n, k, k_extend, w):
        db = self.db.correlation
        return db.find({'n': n, 'w': w, 'k': k, 'k_extend': k_extend}).count() > 0

    def getSolution(self, n, k):
        db = self.db.solution
        db_solution = db.find_one({"n": n, "k": k})
        if db_solution:
            return loads(str(db_solution['class']))
        return []

    def getExtensionOfClass(self, hash):
        db = self.db.classes
        class_bd = db.find_one({'hash': hash})
        if class_bd:
            return class_bd['k_extend']
        return -1