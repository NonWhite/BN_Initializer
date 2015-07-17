from utils import *
from copy import deepcopy as copy
from random import randint as random
from math import log
import os
import sys
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
	
	def hashedarray( self , setfields ) :
		resp = ''
		if not setfields : return resp
		for field in self.data.fields :
			if field not in setfields : continue
			resp += "%s, " % field
		return resp[ :-2 ]
	
#class Learner( Evaluator ) :
	def buildNetwork( self , outfilepath = 'out.txt' ) :
		self.out = open( outfilepath , 'w' )
		self.preprocesscounters()
		network = self.randomSampling()
		self.out.write( "BEST NETWORK:\n" )
		self.printnetwork( network )
		self.saveBestNetwork( network )
	
	def saveBestNetwork( self , network ) :
		best_file = os.path.basename( self.sources[ 0 ] )
		dirname = os.path.dirname( self.sources[ 0 ] )
		best_file = dirname + '/best_' + best_file.replace( 'gentraining_' , '' )
		with open( best_file , 'w' ) as f :
			for field in self.data.fields :
				f.write( "%s:%s\n" % ( field , ', '.join( network[ field ][ 'childs' ] ) ) )
		self.modelfile = best_file
	
	def randomSampling( self ) :
		self.out.write( 'Building network for %s\n' % ','.join( self.sources ) )
		node = { 'parents': [] , 'childs' : [] }
		best_networks = []
		self.entropyvalues = dict( [ ( field , {} ) for field in self.data.fields ] )
		self.sizevalues = dict( [ ( field , {} ) for field in self.data.fields ] )
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
				best_score = ( -INT_MAX if i > 0 else self.bic_score( field , best_parents ) )
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
						current = self.bic_score( field , parents )
						if compare( current , best_score ) > 0 :
							print "BEST_SCORE CHANGED"
							best_score = current
							best_parents = copy( parents )
				self.addRelation( network , field , best_parents , best_score )
			self.printnetwork( network )
			best_networks.append( copy( network ) )
		sorted( best_networks , key = lambda netw : netw[ 'score' ] , reverse = True )
		return best_networks[ 0 ]
	
	def addRelation( self , network , field , parents , score ) :
		network[ field ][ 'parents' ] = copy( parents )
		for p in parents : network[ p ][ 'childs' ].append( field )
		network[ 'score' ] += score
	
	def printnetwork( self , network ) :
		self.out.write( "SCORE = %s\n" % network[ 'score' ] )
		for field in self.data.fields :
			self.out.write( "%s: %s\n" % ( field , ','.join( network[ field ][ 'childs' ] ) ) )
		self.out.write( '\n' )

if __name__ == "__main__" :
	builder = BNBuilder( TRAINING_FILE , savefilter = True , ommit = [ 'fnlgwt' ] )
	builder.addTrainingSet( TEST_FILE )
	builder.loadAndTestModel( DATA_DIR + 'model1.txt' )
	
	'''
	# EXTRACTOR
	sources = TRAINING_FILE
	extractor = Extractor( sources , savefilter = True , ommit = [ 'fnlgwt' ] , discretize = True )
	extractor.printstats()

	# EVALUATOR
	training_data = TRAINING_FILE
	test_data = [ TEST_FILE ]
	models = MODELS
	evaluator = Evaluator( training_data , savefilter = True , ommit = [ 'fnlgwt' ] , discretize = True )
	evaluator.addTrainingSet( test_data )
	for mod in models :
		evaluator.loadAndTestModel( mod )
		evaluator.synthethicData( mod )

	# LEARNER
	# === SYNTHETIC DATA ===
	for i in range( 1 , 4 ) :
		training_data = DATA_DIR + ( 'gentraining_model%s.txt' ) % i
		test_data = DATA_DIR + ( 'gentest_model%s.txt' ) % i
		learner = Learner( training_data )
		output_data = RESULTS_DIR + ( 'gentraining_model%s.txt' ) % i
		learner.buildNetwork( outfilepath = output_data )
		learner.addTrainingSet( [ test_data ] )
		loglikelihood = learner.loadAndTestModel( learner.modelfile )
		learner.out.write( "DATA LOG-LIKELIHOOD = %s" % loglikelihood )
	# === REAL DATA ===
	output_data = RESULTS_DIR + 'realdata.txt'
	learner = Learner( TRAINING_FILE , savefilter = True , ommit = [ 'fnlgwt' ] , discretize = True )
	learner.addTrainingSet( [ TEST_FILE ] )
	learner.buildNetwork( outfilepath = output_data )
	loglikelihood = learner.loadAndTestModel( learner.modelfile )
	learner.out.write( "DATA LOG-LIKELIHOOD = %s" % loglikelihood )
	'''
