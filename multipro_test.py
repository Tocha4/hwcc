from multiprocessing import Process, Queue

def f(q, l):
    print(l)
    x = [42, None, 'hello']
    x.append(l)
    q.put(x)

if __name__ == '__main__':
    
    
    
    
    for i in list('moin'):
        
        q = Queue()
        p = Process(target=f, args=(q,i))
        p.start()
        print(q.get())    # prints "[42, None, 'hello']"
    p.join()