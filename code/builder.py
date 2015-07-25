from utils import *
from copy import deepcopy as copy
from data import Data
from model import Model
import os.path
import sys

class BNBuilder :
	def __init__( self , source , savefilter = False , ommit = [] , discretize = True , outfile = 'out.csv' ) :
		self.data = Data( source , savefilter , ommit , discretize , outfile )
		self.data.calculatecounters()
		self.model = Model( dataobj = self.data )

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

	def greedySearch( self ) :
		self.model.entropyvalues = dict( [ ( field , {} ) for field in self.data.fields ] )
		self.model.sizevalues = dict( [ ( field , {} ) for field in self.data.fields ] )
		self.model.bicvalues = dict( [ ( field , {} ) for field in self.data.fields ] )
		print "Learning bayesian network"
		start = cpu_time()
		init_models = self.initialize()
		best_model = None
		self.out.write( "TIME(initialization) = %s\n" % ( cpu_time() - start ) )
		for i in range( len( init_models ) ) :
			print " ====== INITIAL SOLUTION #%s ======" % (i+1)
			cur_model = copy( init_models[ i ] )
			num_iterations = NUM_GREEDY_ITERATIONS
			for k in xrange( num_iterations ) :
				print " === Iteration #%s === " % (k+1)
				adj_model = self.find_order( cur_model )
				adj_score = adj_model.score()
				cur_score = cur_model.score()
				if compare( adj_score , cur_score ) > 0 :
					cur_model = copy( adj_model )
					print "BEST SCORE = %s" % cur_score
				else :
					num_iterations = k + 1
					break
				self.printnetwork( cur_model.network )
			self.out.write( "TIME(total) = %s\n" % ( cpu_time() - start ) )
			self.out.write( "NUM ITERATIONS = %s\n" % num_iterations )
			if not best_model or compare( cur_model.score() , best_model.score() ) > 0 :
				best_model = cur_model
		return best_model.network

	def dfs( self , graph , node , visited , network ) :
		visited[ node ] = True
		for child in graph[ node ][ 'childs' ] :
			if child in visited : continue
			graph[ node ][ 'childs' ].remove( child )
			graph[ child ][ 'parents' ].remove( node )
			if child not in network[ node ][ 'childs' ] : network[ node ][ 'childs' ].append( child )
			if node not in network[ child ][ 'parents' ]: network[ child ][ 'parents' ].append( node )
			self.dfs( graph , child , visited , network )

	def changedirections( self , graph , network ) :
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
		for i in xrange( len( topo_order ) ) :
			if all_options :
				options = copy( topo_order )
				options.remove( topo_order[ i ] )
			else :
				options = topo_order[ :i ]
			field = topo_order[ i ]
			parents = self.find_best_parents( field , options )
			score = self.model.bic_score( field , parents )
			self.addRelation( network , field, parents , score )
		return network

	def find_order( self , model ) :
		best_model = copy( model )
		order = model.topological
		best_score = INT_MAX
		for i in xrange( len( order ) - 1 ) :
			network = copy( best_model.network )
			self.swap_fields( network , order[ i ] , order[ i + 1 ] )
			self.model.setnetwork( network )
			cur_score = self.model.score()
			if compare( cur_score , best_score ) < 0 :
				best_score = cur_score
				best_model = copy( self.model )
		return best_model
	
	def swap_fields( self , network , field_one , field_two ) :
		for field in self.data.fields :
			if field_one in network[ field ][ 'childs' ] :
				network[ field ][ 'childs' ].remove( field_one )
				network[ field ][ 'childs' ].append( field_two )
				network[ field_one ][ 'parents' ].remove( field )
				network[ field_two ][ 'parents' ].append( field )
			if field_two in network[ field ][ 'childs' ] :
				network[ field ][ 'childs' ].remove( field_two )
				network[ field ][ 'childs' ].append( field_one )
				network[ field_two ][ 'parents' ].remove( field )
				network[ field_one ][ 'parents' ].append( field )

	def find_best_parents( self , field , options ) :
		best_parents = []
		best_score = self.model.bic_score( field , best_parents )
		parents = []
		for tam in xrange( MAX_NUM_PARENTS ) :
			parents.extend( [ list( L ) for L in itertools.combinations( options , tam ) ] )
		for p in parents :
			cur_score = self.model.bic_score( field , p )
			if compare( cur_score , best_score ) < 0 :
				best_score = cur_score
				best_parents = copy( p )
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

	def setInitialSolutionType( self , desc ) :
		if desc == 'random' : self.initialize = self.random_solutions
		elif desc == 'unweighted' : self.initialize = self.unweighted_solutions
		elif desc == 'weighted' : self.initialize = self.weighted_solutions

	def random_solutions( self ) :
		solutions = []
		num_fields = len( self.data.fields )
		model = copy( self.model )
		for k in range( num_fields ) :
			order = shuffle( self.data.fields )
			network = self.find_greedy_network( order , all_options = False )
			model.setnetwork( network , topo_order = order )
			solutions.append( model )
		return solutions

	# TODO: Test this
	def unweighted_solutions( self ) :
		print "Building graph with best parents for each field"
		greedy_graph = self.find_greedy_network( self.data.fields , all_options = True )
		print "GREEDY GRAPH"
		for f in self.data.fields : print "%s: %s" % ( f , greedy_graph[ f ][ 'parents' ] )
		print "Getting feedback arc set for each node"
		solutions = []
		for field in self.data.fields :
			print " === Building network with root %s === " % field.upper()
			network = self.clean_graph()
			visited = {}
			graph = copy( greedy_graph )
			self.dfs( graph , field , visited , network )
			print "Changing edge's directions that made cycles"
			self.changedirections( graph , network )
			#print "NETWORK"
			#for f in self.data.fields : print "%s: %s" % ( f , network[ f ][ 'childs' ] )
			model = copy( self.model )
			model.setnetwork( network )
			solutions.append( model )
		return solutions

	# TODO: Implement this
	def weighted_solutions( self ) :
		return [ '' ] * 10
	
if __name__ == "__main__" :
	if len( sys.argv ) == 4 :
		dataset_file , ommit_fields , out_file = sys.argv[ 1: ]
		ommit_fields = [ f.strip() for f in ommit_fields.split( ',' ) ]
		builder = BNBuilder( dataset_file , savefilter = True , ommit = ommit_fields )

		print "========== RUNNING WITH RANDOM PERMUTATION =========="
		builder.setInitialSolutionType( 'random' )
		builder.buildNetwork( outfilepath = out_file % 'random' )
		
		print "========== RUNNING WITH DFS =========="
		builder.setInitialSolutionType( 'unweighted' )
		builder.buildNetwork( outfilepath = out_file % 'unweighted' )
		
		print "========== RUNNING WITH MST + DFS =========="
		builder.setInitialSolutionType( 'weighted' )
		builder.buildNetwork( outfilepath = out_file % 'weighted' )
