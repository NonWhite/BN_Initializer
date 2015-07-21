from utils import *
from copy import deepcopy as copy
import os.path
from data import Data
from model import Model

class BNBuilder :
	def __init__( self , source , savefilter = False , ommit = [] , discretize = True , outfile = 'out.csv' , initialrandom = True ) :
		self.data = Data( source , savefilter , ommit , discretize , outfile )
		self.data.calculatecounters()
		self.model = Model( dataobj = self.data )
		self.initialrandom = initialrandom

	def addTrainingSet( self , testfile ) :
		self.model.addtrainingset( testfile , self.data.ommitedfields )

	def loadAndTestModel( self , modelfile ) :
		self.model.loaddata( self.data )
		self.model.loadmodel( modelfile )
		self.model.trainmodel()
		return self.model.testmodel()

	def buildNetwork( self , outfilepath = 'out.txt' ) :
		self.out = open( outfilepath , 'w' )
		#network = self.randomSampling()
		network = self.greedySearch()
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

	''' TODO: Test this '''
	def greedySearch( self ) :
		self.model.entropyvalues = dict( [ ( field , {} ) for field in self.data.fields ] )
		self.model.sizevalues = dict( [ ( field , {} ) for field in self.data.fields ] )
		best_model = self.getinitialorder()
		print " ================================================================================================================================= "
		for k in range( NUM_GREEDY_ITERATIONS ) :
			print "Iteration #%s" % (k+1)
			cur_model = self.find_order( best_model )
			if compare( cur_model.score() , best_model.score()  ) > 0 :
				best_model = copy( cur_model )
		return best_model.network

	''' TODO: Test this '''
	def getinitialorder( self ) :
		model = copy( self.model )
		if self.initialrandom :
			order = shuffle( self.data.fields )
			network = self.find_greedy_network( order , all_options = False )
			model.setnetwork( network , topo_order = order )
		else :
			print "Building graph with best parents for each field"
			greedy_graph = self.find_greedy_network( self.data.fields , all_options = True )
			best_score = -INT_MAX
			print "Feedback arc set for each node"
			for field in self.data.fields :
				print "Building network with root = %s" % field
				network = self.clean_graph()
				visited = {}
				graph = copy( greedy_graph )
				print "GRAPH"
				for f in self.data.fields : print "%s: %s" % ( f , graph[ f ][ 'childs' ] )
				self.dfs( graph , field , visited , network )
				print "NETWORK"
				for f in self.data.fields : print "%s: %s" % ( f , network[ f ][ 'childs' ] )
				print "Changing some edges directions"
				self.change( graph , network )
				self.model.setnetwork( network )
				print "Calculating BIC Score for network"
				cur_score = self.model.score()
				if compare( cur_score , best_score ) > 0 :
					best_score = cur_score
					model = copy( self.model() )
		return model
	
	def dfs( self , graph , node , visited , network ) :
		visited[ node ] = True
		for child in graph[ node ][ 'childs' ] :
			if child in visited : continue
			graph[ node ][ 'childs' ].remove( child )
			graph[ child ][ 'parents' ].remove( node )
			if child not in network[ node ][ 'childs' ] : network[ node ][ 'childs' ].append( child )
			if node not in network[ child ][ 'parents' ]: network[ child ][ 'parents' ].append( node )
			self.dfs( graph , child , visited , network )
	
	def change( self , graph , network ) :
		for field in self.data.fields :
			if field not in graph : continue
			for par in graph[ field ][ 'parents' ] :
				if field not in network[ par ][ 'parents' ] : network[ par ][ 'parents' ].append( field )
				if par not in network[ field ][ 'childs' ] : network[ field ][ 'childs' ].append( par )
			for ch in graph[ field ][ 'childs' ] :
				if field not in network[ ch ][ 'childs' ] : network[ ch ][ 'childs' ].append( field )
				if ch not in network[ field ][ 'parents' ] : network[ field ][ 'parents' ].append( ch )
	
	def clean_graph( self ) :
		node = { 'parents': [] , 'childs' : [] }
		network = dict( [ ( field , copy( node ) ) for field in self.data.fields ] )
		network[ 'score' ] = 0.0
		return network
	
	def find_greedy_network( self , topo_order , all_options = False ) :
		network = self.clean_graph()
		for i in range( len( topo_order ) ) :
			if all_options :
				options = copy( self.data.fields )
				options.remove( self.data.fields[ i ] )
			else :
				options = topo_order[ :i ]
			field = topo_order[ i ]
			parents = self.find_best_parents( field , options )
			score = self.model.bic_score( field , parents )
			self.addRelation( network , field, parents , score )
		return network

	def find_order( self , model ) :
		best = copy( model )
		order = model.topological
		score = model.score()
		for i in range( len( order ) - 1 ) :
			new_order = copy( order )
			new_order[ i ] , new_order[ i + 1 ] = new_order[ i + 1 ] , new_order[ i ]
			network = self.find_greedy_network( new_order )
			self.model.setnetwork( network )
			cur_score = self.model.score()
			if compare( cur_score , score ) > 0 :
				score = cur_score
				best = copy( self.model )
		return best

	def find_best_parents( self , field , options ) :
		best_parents = []
		best_score = self.model.bic_score( field , best_parents )
		for p in range( MAX_NUM_PARENTS ) :
			opt_parents = []
			opt_score = -INT_MAX
			for opt in options :
				cur_parents = copy( best_parents )
				cur_parents.append( opt )
				cur_score = self.model.bic_score( field , cur_parents )
				if compare( cur_score , opt_score ) > 0 :
					opt_score = cur_score
					opt_parents = copy( cur_parents )
			if compare( opt_score , best_score ) > 0 :
				best_score = opt_score
				best_parents = copy( opt_parents )
				options.remove( best_parents[ -1 ] )
			else :
				break
		return best_parents

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
	builder = BNBuilder( TRAINING_FILE , savefilter = True , ommit = [ 'fnlgwt' ] , initialrandom = False )
	#builder.loadAndTestModel( DATA_DIR + 'model1.txt' )
	outfile = RESULTS_DIR + 'test.txt'
	builder.buildNetwork( outfilepath = outfile )
	builder.addTrainingSet( TEST_FILE )
	loglikelihood = builder.loadAndTestModel( builder.modelfile )
	builder.out.write( "DATA LOG-LIKELIHOOD = %s" % loglikelihood )
