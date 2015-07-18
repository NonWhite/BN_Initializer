from utils import *
from copy import deepcopy as copy
import os.path
from data import Data
from model import Model

class BNBuilder :
	def __init__( self , source , savefilter = False , ommit = [] , discretize = True , outfile = 'out.csv' ) :
		self.data = Data( source , savefilter , ommit , discretize , outfile )
		self.data.calculatecounters()
		self.model = Model()
	
	def addTrainingSet( self , testfile ) :
		self.model.addtrainingset( testfile , self.data.ommitedfields )
	
	def loadAndTestModel( self , modelfile ) :
		self.model.loaddata( self.data )
		self.model.loadmodel( modelfile )
		self.model.trainmodel()
		return self.model.testmodel()
	
	def buildNetwork( self , outfilepath = 'out.txt' ) :
		self.out = open( outfilepath , 'w' )
		network = self.randomSampling()
		#network = self.greedySearch()
		self.out.write( "BEST NETWORK:\n" )
		self.printnetwork( network )
		self.saveBestNetwork( network )
	
	def saveBestNetwork( self , network ) :
		best_file = os.path.basename( self.data.source )
		dirname = os.path.dirname( self.data.source )
		best_file = dirname + '/best_' + best_file.replace( 'gentraining_' , '' )
		with open( best_file , 'w' ) as f :
			for field in self.data.fields :
				f.write( "%s:%s\n" % ( field , ', '.join( network[ field ][ 'childs' ] ) ) )
		self.modelfile = best_file
	
	def randomSampling( self ) :
		self.out.write( 'Building network for %s\n' % self.data.source )
		node = { 'parents': [] , 'childs' : [] }
		best_networks = []
		self.model = Model( self.data )
		self.model.entropyvalues = dict( [ ( field , {} ) for field in self.data.fields ] )
		self.model.sizevalues = dict( [ ( field , {} ) for field in self.data.fields ] )
		for k in range( NUM_ORDERING_SAMPLES ) :
			lst_fields = shuffle( self.data.fields )
			network = dict( [ ( field , copy( node ) ) for field in self.data.fields ] )
			network[ 'score' ] = 0.0
			self.out.write( "Building network #%s\n" % ( k + 1 ) )
			print "Building network #%s" % ( k + 1 )
			for i in range( len( lst_fields ) ) :
				field = lst_fields[ i ]
				print "======== Field #%s: %s ========" % ( i , field )
				best_parents = []
				best_score = ( -INT_MAX if i > 0 else self.model.bic_score( field , best_parents ) )
				for t in range( NUM_GREEDY_RESTARTS ) :
					if i == 0 : break # First field doesn't have parents
					parents = []
					max_num_parents = min( MAX_NUM_PARENTS , i )
					for n in range( max_num_parents ) :
						while True :
							pos = randint( 0 , i - 1 )
							new_parent = lst_fields[ pos ]
							if new_parent not in parents : break
						parents.append( new_parent )
						current = self.model.bic_score( field , parents )
						if compare( current , best_score ) > 0 :
							#print "BEST SCORE CHANGED"
							best_score = current
							best_parents = copy( parents )
				self.addRelation( network , field , best_parents , best_score )
			self.printnetwork( network )
			best_networks.append( copy( network ) )
		sorted( best_networks , key = lambda netw : netw[ 'score' ] , reverse = True )
		return best_networks[ 0 ]
	
	# TODO: Test
	def greedySearch( self ) :
		best_order , best_network = self.getinitialorder()
		for k in range( NUM_ORDERINGS ) :
			cur_order , cur_network = self.find_order( best_order )
			if compare( self.score( cur_network ) , self.score( best_order ) ) > 0 :
				best_order = copy( cur_order )
				best_network = copy( cur_network )
		return best_network
	
	# TODO
	def getinitialorder( self ) :
		return self.data.fields
	
	#TODO
	def find_order( self , order ) :
		return order
	
	# TODO
	def score( self , network ) :
		return 0.0
	
	def addRelation( self , network , field , parents , score ) :
		network[ field ][ 'parents' ] = copy( parents )
		for p in parents : network[ p ][ 'childs' ].append( field )
		network[ 'score' ] += score
	
	def printnetwork( self , network ) :
		self.out.write( "SCORE = %s\n" % network[ 'score' ] )
		for field in self.data.fields :
			self.out.write( "%s: %s\n" % ( field , ','.join( network[ field ][ 'childs' ] ) ) )
		self.out.write( '\n' )
		self.out.flush()

if __name__ == "__main__" :
	builder = BNBuilder( TRAINING_FILE , savefilter = True , ommit = [ 'fnlgwt' ] )
	#builder.loadAndTestModel( DATA_DIR + 'model1.txt' )
	'''
	outfile = RESULTS_DIR + 'test.txt'
	builder.buildNetwork( outfilepath = outfile )
	builder.addTrainingSet( TEST_FILE )
	loglikelihood = builder.loadAndTestModel( builder.modelfile )
	builder.out.write( "DATA LOG-LIKELIHOOD = %s" % loglikelihood )
	'''
