class no():
    def __init__(self, dado, prox = None):
        self.prox = prox
        self.dado = dado
class lista_encadeada():
    def __init__(self, cabeca, tamanho):
        self.cabeca = cabeca
        self.tamanho = tamanho
    def adicionar(self, dado, index):
        if(self.tamanho == 0):
            self.cabeca = no(dado)
            self.cabeca.prox = None
            self.tamanho = self.tamanho + 1
            return
        elif(index == 0 and self.tamanho != 0):
            temporario = self.cabeca
            self.cabeca.dado = dado
            self.cabeca.prox = temporario
            temporario.prox = None
            self.tamanho = self.tamanho + 1
            return
        else:
            celtemp = self.cabeca
            for i in range(index-1):
                celtemp = self.cabeca.prox
                if(celtemp.prox == None):
                    break
            cel2 = celtemp.prox
            celtemp.prox = no(dado)
            celtemp.prox.prox = cel2
            self.tamanho = self.tamanho + 1
            return
    def remover(self, index):
        if(index == 0 and self.tamanho == 1):
            self.cabeca = None
            self.tamanho = 0
        elif(index == 0 and self.tamanho>1):
            self.cabeca = self.cabeca.prox
            self.tamanho = self.tamanho-1
        else:
            for i in range(index-1):
                celtemp = self.cabeca.prox
                if(celtemp.prox == None):
                    break
            cel2 = celtemp.prox
            celtemp.prox = cel2.prox
            cel2.prox = None
            cel2 = None
            self.tamanho = self.tamanho - 1
            return
    def imprimir(self):
        celula = self.cabeca
        for i in range(self.tamanho):
            print(str(celula.dado))
            celula = celula.prox
        return
        
lista_encadeada = lista_encadeada(None, 0)
lista_encadeada.adicionar(0, 0)
lista_encadeada.adicionar(1, 1)
lista_encadeada.adicionar(2, 2)
lista_encadeada.remover(2)
lista_encadeada.imprimir()

                
                
            
            
        
        
        