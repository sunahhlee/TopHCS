import unittest
import csvec
from csvec import CSVec
from topHCS import TopHCS
import torch

class Base:
    # use Base class to hide TestCase from the unittest runner
    # we only want the subclasses to actually be run

    class TopHCSTestCase(unittest.TestCase):
        def testInit(self):
            d = 100
            c = 20
            r = 5
            h = 36

            a = TopHCS(h, d, c, r, **self.csvecArgs)
            zerosTable = torch.zeros(r, c).to(self.device)
            self.assertTrue(torch.allclose(a.csvec.table, zerosTable))

            zerosH = torch.zeros(h).to(self.device)
            self.assertTrue(torch.allclose(a.topH, zerosH))
            self.assertTrue(a.h == h)

        def testStoreVec1(self):
            # store vec [1, 100], recover 1
            d = 100
            c = 20
            r = 5
            h = 1
            
            a = TopHCS(h, d, c, r, **self.csvecArgs)
            vec = torch.arange(1, d+1, dtype=torch.float, device=self.device)
            a.store(vec)
            
            expected = torch.zeros(d, dtype=torch.float, device=self.device)
            expected[d-1] = d

            self.assertTrue(torch.allclose(a.topH, expected))
            self.assertTrue(torch.allclose(a.bottomH, vec - expected))
            
        def testStoreVec2(self):
            # store vec [1, 100], recover all
            d = 100
            c = 20
            r = 5
            h = d
            
            a = TopHCS(h, d, c, r, **self.csvecArgs)
            vec = torch.arange(1, d+1, dtype=torch.float, device=self.device)
            a.store(vec)

            self.assertTrue(torch.allclose(a.topH, vec))
            
        def testStoreVec3(self):
            # store randn tensor, recover all
            d = 100
            c = 20
            r = 5
            h = d
            
            a = TopHCS(h, d, c, r, **self.csvecArgs)
            vec = torch.randn(d)
            a.store(vec)

            self.assertTrue(torch.allclose(a.topH, vec))
            
        def testSameBuckets(self):
            d = 100
            c = 20
            r = 5
            h = 0
            a = CSVec(d, c, r, **self.csvecArgs)
            vec = torch.randn(d)
            a += vec
            b = TopHCS(h=h, d=d, c=c, r=r, **self.csvecArgs)
            b.store(vec)
            self.assertTrue(torch.allclose(a.table, b.csvec.table))
        def testTopKSum(self):
            d = 10
            c = 10000
            r = 20
            h = d

            a = TopHCS(h, d, c, r, **self.csvecArgs) 
            b = TopHCS(h, d, c, r, **self.csvecArgs) 
            zerosHalf = torch.zeros(d//2, dtype=torch.float, device=self.device) 
            vec = torch.cat((torch.randn(d//2, device=self.device), zerosHalf), 0)
            vec2 = torch.cat((zerosHalf, torch.randn(d//2, device=self.device)), 0)
            a.store(vec)
            b.store(vec2)

            result = TopHCS.topKSum([a, b], d) 
            expected = vec + vec2
            self.assertTrue(torch.equal(expected, result))

        def testTopKSum2(self):
            d = 10
            c = 10000
            r = 20
            h = d

            a = TopHCS(h, d, c, r, **self.csvecArgs) 
            b = TopHCS(h, d, c, r, **self.csvecArgs) 
            c = TopHCS(h, d, c, r, **self.csvecArgs) 
            vec = torch.randn(d, device=self.device)
            vec2 = torch.randn(d, device=self.device)
            vec3 = torch.randn(d, device=self.device)
            a.store(vec)
            b.store(vec2)
            c.store(vec3)

            result = TopHCS.topKSum([a, b, c], d) 
            expected = vec + vec2 + vec3
            self.assertTrue(torch.equal(expected, result))

class TestCaseCPU1(Base.TopHCSTestCase):
    def setUp(self):
        # hack to reset csvec's global cache between tests
        csvec.cache = {}

        self.device = "cpu"
        self.numBlocks = 1

        self.csvecArgs = {"numBlocks": self.numBlocks,
                          "device": self.device}

class TestCaseCPU2(Base.TopHCSTestCase):
    def setUp(self):
        csvec.cache = {}

        self.device = "cpu"
        self.numBlocks = 2

        self.csvecArgs = {"numBlocks": self.numBlocks,
                          "device": self.device}

@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestCaseCUDA2(Base.TopHCSTestCase):
    def setUp(self):
        csvec.cache = {}

        self.device = "cuda"
        self.numBlocks = 2

        self.csvecArgs = {"numBlocks": self.numBlocks,
                          "device": self.device}
