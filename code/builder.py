from utils import *
from random import randint
from copy import deepcopy as copy
from unionfind import UnionFind
from data import Data
from model import Model
import os.path
import sys

class BNBuilder :
	def __init__( self , source , savefilter = False , ommit = [] , discretize = True ) :
		outfile = RESULTS_DIR + os.path.basename( source )
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
		network = self.greedySearch()
		self.out.write( "BEST NETWORK:\n" )
		self.printnetwork( network )
		self.saveBestNetwork( network )

	def saveBestNetwork( self , network ) :
		dirname = os.path.dirname( self.data.source )
		best_file = "%s/best_%s" % ( dirname , os.path.basename( self.data.source ) )
		with open( best_file , 'w' ) as f :
			for field in self.data.fields :
				f.write( "%s:%s\n" % ( field , ', '.join( network[ field ][ 'childs' ] ) ) )
		self.modelfile = best_file

	def greedySearch( self ) :
		self.model.initdict()
		print "Learning bayesian network from dataset %s" % self.data.source
		start = cpu_time()
		init_orders = self.initialize()
		best_order = None
		self.out.write( "TIME(initialization) = %s\n" % ( cpu_time() - start ) )
		for i in xrange( len( init_orders ) ) :
			print " ====== INITIAL SOLUTION #%s ======" % (i+1)
			self.out.write( " ====== INITIAL SOLUTION #%s ======\n" % (i+1) )
			cur_order = copy( init_orders[ i ] )
			num_iterations = NUM_GREEDY_ITERATIONS
			for k in xrange( num_iterations ) :
				print " === Iteration #%s === " % (k+1)
				adj_order = self.find_order( cur_order )
				restart_order = self.random_restart( adj_order )
				if self.better_order( restart_order , adj_order ) :
					adj_order = copy( restart_order )
				if self.better_order( adj_order , cur_order ) :
					cur_order = copy( adj_order )
					self.model.setnetwork( self.find_greedy_network( cur_order ) , train = False )
					score = self.model.score()
					print "BEST SCORE = %s" % score
					self.printnetwork( self.model.network )
				else :
					self.model.setnetwork( self.find_greedy_network( cur_order ) , train = False )
					score = self.model.score()
					print "SOLUTION CONVERGES to %s" % score
					num_iterations = k + 1
					self.printnetwork( self.model.network )
					break
			self.out.write( "TIME(total) = %s\n" % ( cpu_time() - start ) )
			self.out.write( "NUM ITERATIONS = %s\n" % num_iterations )
			if not best_order or self.better_order( cur_order , best_order ) :
				best_order = copy( cur_order )
		best_network = self.find_greedy_network( best_order )
		return best_network

	def random_restart( self , order ) :
		for i in xrange( NUM_RANDOM_RESTARTS ) :
			p1 = randint( 0 , len( order ) - 1 )
			p2 = randint( 0 , len( order ) - 1 )
			order = self.swap_fields( order , p1 , p2 )
		return order

	def better_order( self , order1 , order2 ) :
		model = copy( self.model )
		net_1 = self.find_greedy_network( order1 )
		model.setnetwork( net_1 , train = True )
		score1 = model.score()
		net_2 = self.find_greedy_network( order2 )
		model.setnetwork( net_2 , train = True )
		score2 = model.score()
		return self.isbetter( score1 , score2 )

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

	def find_order( self , order ) :
		best_order = copy( order )
		best_score = self.worst_score_value()
		for i in xrange( len( order ) - 1 ) :
			new_order = self.swap_fields( order , i , i + 1 )
			network = self.find_greedy_network( new_order )
			cur_model = copy( self.model )
			cur_model.setnetwork( network , train = True )
			cur_score = cur_model.score()
			if self.isbetter( cur_score , best_score ) :
				best_score = copy( cur_score )
				best_model = copy( cur_model )
		return best_model.topological

	def swap_fields( self , order , idx1 , idx2 ) :
		new_order = copy( order )
		new_order[ idx1 ] , new_order[ idx2 ] = new_order[ idx2 ] , new_order[ idx1 ]
		return new_order

	def find_best_parents( self , field , options ) :
		best_parents = []
		best_score = self.worst_score_value()
		possible_sets = []
		for tam in xrange( MAX_NUM_PARENTS ) :
			possible_sets.extend( [ list( L ) for L in itertools.combinations( options , tam ) ] )
		for p in possible_sets :
			cur_score = self.model.bic_score( field , p )
			if self.isbetter( cur_score , best_score ) :
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

	def isbetter( self , score1 , score2 ) :
		resp = compare( score1 , score2 )
		return resp > 0

	def worst_score_value( self ) :
		return -INT_MAX

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
			network = self.find_greedy_network( order )
			model.setnetwork( network , topo_order = order , train = False )
			solutions.append( model.topological )
		return solutions

	def dfs( self , graph , node , visited , network ) :
		visited[ node ] = True
		for child in graph[ node ][ 'childs' ] :
			if child in visited : continue
			''' Deleting in graph '''
			graph[ node ][ 'childs' ].remove( child )
			graph[ child ][ 'parents' ].remove( node )
			''' Adding to network '''
			network[ node ][ 'childs' ].append( child )
			network[ child ][ 'parents' ].append( node )
			self.dfs( graph , child , visited , network )
	
	def change_directions( self , graph , network ) :
		for field in self.data.fields :
			if field not in graph : continue
			for par in graph[ field ][ 'parents' ] :
				self.add_edge( network , field , par )
			for ch in graph[ field ][ 'childs' ] :
				self.add_edge( network , ch , field )
	
	def add_edge( self , graph , from_node , to_node ) :
		if from_node in graph[ to_node ][ 'childs' ] : return
		if to_node in graph[ from_node ][ 'parents' ] : return
		if to_node not in graph[ from_node ][ 'childs' ] :
			graph[ from_node ][ 'childs' ].append( to_node )
		if from_node not in graph[ to_node ][ 'parents' ] :
			graph[ to_node ][ 'parents' ].append( from_node )

	def unweighted_solutions( self ) :
		print "Building graph with best parents for each field"
		greedy_graph = self.find_greedy_network( self.data.fields , all_options = True )
		print "GREEDY GRAPH"
		for f in self.data.fields : print "%s:%s" % ( f , greedy_graph[ f ][ 'parents' ] )
		solutions = []
		for field in self.data.fields :
			print " === Building network with root %s === " % field.upper()
			network = self.clean_graph()
			visited = {}
			graph = copy( greedy_graph )
			self.dfs( graph , field , visited , network )
			self.change_directions( graph , network )
			model = copy( self.model )
			model.setnetwork( network , train = False )
			print ', '.join( model.topological )
			solutions.append( model.topological )
		return solutions

	def add_weights( self , graph ) :
		G = self.clean_graph()
		for field in self.data.fields :
			for child in graph[ field ][ 'childs' ] :
				weight = self.model.bic_score( field , graph[ field ][ 'parents' ] ) - \
							self.model.bic_score( field , [ child ] )
				G[ field ][ 'childs' ].append( ( child , weight ) )
				G[ child ][ 'parents' ].append( ( field , weight ) )
				G[ field ][ 'parents' ].append( ( child , weight ) )
				G[ child ][ 'childs' ].append( ( field , weight ) )
		return G

	def kruskal( self , graph ) :
		p = UnionFind( len( self.data.fields ) )
		edges = []
		for field in self.data.fields :
			id1 = self.data.fields.index( field )
			for ( child , weight ) in graph[ field ][ 'childs' ] :
				id2 = self.data.fields.index( child )
				edges.append( ( weight , id1 , id2 , field , child ) )
				edges.append( ( weight , id2 , id2 , child , field ) )
		sorted( edges , key = lambda edg : edg[ 0 ] )
		M = self.clean_graph()
		for edg in edges :
			x , y , node1 , node2 = edg[ 1: ]
			if p.sameSet( x , y ) : continue
			p.unionSet( x , y )
			M[ node1 ][ 'childs' ].append( node2 )
			M[ node2 ][ 'parents' ].append( node1 )
			M[ node2 ][ 'childs' ].append( node1 )
			M[ node1 ][ 'parents' ].append( node2 )
		return M

	def weighted_solutions( self ) :
		print "Building graph with best parents for each field"
		greedy_graph = self.find_greedy_network( self.data.fields , all_options = True )
		print "GREEDY GRAPH"
		for f in self.data.fields : print "%s:%s" % ( f , greedy_graph[ f ][ 'parents' ] )
		kruskal_graph = self.add_weights( greedy_graph )
		mst = self.kruskal( kruskal_graph )
		solutions = []
		for field in self.data.fields :
			print " === Building network with root %s === " % field.upper()
			network = self.clean_graph()
			visited = {}
			graph = copy( mst )
			self.dfs( graph , field , visited , network )
			self.change_directions( graph , network )
			model = copy( self.model )
			model.setnetwork( network , train = False )
			print ', '.join( model.topological )
			solutions.append( model.topological )
		return solutions
	
if __name__ == "__main__" :
	if len( sys.argv ) == 4 :
		dataset_file , ommit_fields , out_file = sys.argv[ 1: ]
		if ommit_fields == 'None' : ommit_fields = []
		else : ommit_fields = [ f.strip() for f in ommit_fields.split( ',' ) ]
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
