from utils import *
from random import randint
from copy import deepcopy as copy
from data import Data
from model import Model
import os.path
import sys

class BNBuilder :
	def __init__( self , source , savefilter = False , ommit = [] , discretize = False ) :
		outfile = RESULTS_DIR + os.path.basename( source )
		self.dataset_name = os.path.splitext( os.path.basename( source ) )[ 0 ]
		self.data = Data( source , savefilter , ommit , discretize , outfile )
		self.data.calculatecounters()
		self.model = Model( dataobj = self.data )
		self.best_parents = dict( [ ( field , {} ) for field in self.data.fields ] )

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
		self.printnetwork( network , printrelations = True )
		#self.saveBestNetwork( network )

	def saveBestNetwork( self , network ) :
		dirname = os.path.dirname( self.data.source )
		best_file = "%s/best_%s" % ( dirname , os.path.basename( self.data.source ) )
		with open( best_file , 'w' ) as f :
			for field in self.data.fields :
				f.write( "%s:%s\n" % ( field , ', '.join( network[ field ][ 'childs' ] ) ) )
		self.modelfile = best_file

	def greedySearch( self ) :
		#self.model.initialize()
		self.model.setnetwork( self.clean_graph() , train = False )
		self.base_score = self.model.score()
		print "Learning bayesian network from dataset %s" % self.data.source
		init_orders = self.initialize()
		best_order = None
		''' START POINTER FUNCTIONS '''
		better_order = self.better_order
		setnetwork = self.model.setnetwork
		networkscore = self.model.score
		printnetwork = self.printnetwork
		write = self.out.write
		''' END POINTER FUNCTIONS '''
		for i in xrange( len( init_orders ) ) :
			print " ============ INITIAL SOLUTION #%s ============" % (i+1)
			write( " ============ INITIAL SOLUTION #%s ============\n" % (i+1) )
			cur_order = copy( init_orders[ i ] )
			print cur_order
			num_iterations = NUM_GREEDY_ITERATIONS
			for k in xrange( num_iterations ) :
				#print " ====== Iteration #%s ====== " % (k+1)
				adj_order = self.find_order( cur_order )
				restart_order = self.random_restart( adj_order )
				if better_order( restart_order , adj_order ) :
					adj_order = copy( restart_order )
				if better_order( adj_order , cur_order ) :
					cur_order = copy( adj_order )
					setnetwork( self.find_greedy_network( cur_order ) , train = False )
					score = networkscore()
					#print "BEST SCORE = %s" % score
					printnetwork( self.model.network )
				else :
					setnetwork( self.find_greedy_network( cur_order ) , train = False )
					score = networkscore()
					#print "SOLUTION CONVERGES to %s" % score
					num_iterations = k + 1
					printnetwork( self.model.network )
					break
			write( "NUM ITERATIONS = %s\n" % num_iterations )
			if not best_order or better_order( cur_order , best_order ) :
				best_order = copy( cur_order )
		best_network = self.find_greedy_network( best_order )
		return best_network

	def random_restart( self , order ) :
		''' START POINTER FUNCTIONS '''
		swap = self.swap_fields
		''' END POINTER FUNCTIONS '''
		for i in xrange( NUM_RANDOM_RESTARTS ) :
			p1 = randint( 0 , len( order ) - 1 )
			p2 = randint( 0 , len( order ) - 1 )
			order = swap( order , p1 , p2 )
		return order

	def better_order( self , order1 , order2 ) :
		net_1 = self.find_greedy_network( order1 )
		self.model.setnetwork( net_1 , train = True )
		score1 = self.model.score()
		net_2 = self.find_greedy_network( order2 )
		self.model.setnetwork( net_2 , train = True )
		score2 = self.model.score()
		return self.isbetter( score1 , score2 )

	def clean_graph( self ) :
		node = { 'parents': [] , 'childs' : [] }
		network = dict( [ ( field , copy( node ) ) for field in self.data.fields ] )
		network[ 'score' ] = 0.0
		return network

	def find_greedy_network( self , topo_order , all_options = False ) :
		network = self.clean_graph()
		''' START POINTER FUNCTIONS '''
		find_best_parents = self.find_best_parents
		bic_score = self.model.bic_score
		add_relation = self.addRelation
		''' END POINTER FUNCTIONS '''
		for i in xrange( len( topo_order ) ) :
			if all_options :
				options = copy( topo_order )
				options.remove( topo_order[ i ] )
			else :
				options = topo_order[ :i ]
			field = topo_order[ i ]
			parents = find_best_parents( field , options )
			score = bic_score( field , parents )
			add_relation( network , field, parents , score )
		return network

	def find_order( self , order ) :
		''' START POINTER FUNCTIONS '''
		swap = self.swap_fields
		find_greedy_network = self.find_greedy_network
		networkscore = self.model.score
		setnetwork = self.model.setnetwork
		''' END POINTER FUNCTIONS '''
		best_order = copy( order )
		best_score = self.worst_score_value()
		for i in xrange( len( order ) - 1 ) :
			new_order = self.swap_fields( order , i , i + 1 )
			network = find_greedy_network( new_order )
			setnetwork( network , train = True )
			cur_score = networkscore()
			if self.isbetter( cur_score , best_score ) :
				best_score = copy( cur_score )
				best_model = copy( self.model )
		return best_model.topological

	def swap_fields( self , order , idx1 , idx2 ) :
		new_order = copy( order )
		new_order[ idx1 ] , new_order[ idx2 ] = new_order[ idx2 ] , new_order[ idx1 ]
		return new_order

	def find_best_parents( self , field , options ) :
		# Verify if it has been already calculated
		hash_parents = self.model.hashedarray( options )
		if hash_parents in self.best_parents[ field ] : return self.best_parents[ field ][ hash_parents ]
		lstparents = self.model.bestparents[ field ]
		pos = [ idx for idx in xrange( len( lstparents ) ) if set( lstparents[ idx ] ).issubset( options ) ]
		best = []
		if pos : best = lstparents[ pos[ 0 ] ]
		self.best_parents[ field ][ hash_parents ] = best
		return best

	def addRelation( self , network , field , parents , score ) :
		network[ field ][ 'parents' ] = copy( parents )
		for p in parents : network[ p ][ 'childs' ].append( field )
		network[ 'score' ] += score

	def printnetwork( self , network , printrelations = False ) :
		''' START POINTER FUNCTIONS '''
		write = self.out.write
		''' END POINTER FUNCTIONS '''
		write( "SCORE = %s\n" % network[ 'score' ] )
		if printrelations :
			for field in self.data.fields :
				write( "%s: %s\n" % ( field , ','.join( network[ field ][ 'childs' ] ) ) )

	def isbetter( self , score1 , score2 ) :
		resp = compare( score1 , score2 )
		return resp > 0

	def worst_score_value( self ) :
		return -INT_MAX

	def setInitialSolutionType( self , desc ) :
		if desc == 'random' : self.initialize = self.random_solutions
		elif desc == 'unweighted' : self.initialize = self.unweighted_solutions
		elif desc == 'weighted' : self.initialize = self.weighted_solutions

	''' =========================== RANDOM SOLUTION APPROACH =========================== '''
	def random_solutions( self ) :
		solutions = []
		''' START POINTER FUNCTIONS '''
		setnetwork = self.model.setnetwork
		find_greedy_network = self.find_greedy_network
		append = solutions.append
		lstfields = self.data.fields
		''' END POINTER FUNCTIONS '''
		for k in xrange( NUM_INITIAL_SOLUTIONS ) :
			order = shuffle( lstfields )
			network = find_greedy_network( order )
			setnetwork( network , topo_order = order , train = False )
			append( copy( self.model.topological ) )
		return solutions
	
	''' =========================== DFS APPROACH =========================== '''
	def dfs( self , graph , node , unvisited , order ) :
		unvisited.remove( node )
		order.append( node )
		graph[ node ][ 'childs' ] = shuffle( graph[ node ][ 'childs' ] )
		''' START POINTER FUNCTIONS '''
		dfs = self.dfs
		''' END POINTER FUNCTIONS '''
		for child in graph[ node ][ 'childs' ] :
			if child not in unvisited : continue
			dfs( graph , child , unvisited , order )
	
	def traverse_graph( self , graph ) :
		''' START POINTER FUNCTIONS '''
		dfs = self.dfs
		lstfields = self.data.fields
		''' END POINTER FUNCTIONS '''
		unvisited = copy( lstfields )
		G = copy( graph )
		order = []
		length = len( lstfields )
		while unvisited :
			pos = randint( 0 , len( unvisited ) - 1 )
			root = unvisited[ pos ]
			dfs( G , root , unvisited , order )
		return order

	def unweighted_solutions( self ) :
		print "Building graph with best parents for each field"
		greedy_graph = self.find_greedy_network( self.data.fields , all_options = True )
		print "GREEDY GRAPH"
		for f in self.data.fields : print "%s:%s" % ( f , greedy_graph[ f ][ 'parents' ] )
		solutions = []
		''' START POINTER FUNCTIONS '''
		traverse = self.traverse_graph
		append = solutions.append
		''' END POINTER FUNCTIONS '''
		for i in xrange( NUM_INITIAL_SOLUTIONS ) :
			print " === Building network #%s === " % (i+1)
			order = traverse( greedy_graph )
			append( copy( order ) )
		return solutions

	''' =========================== FAS APPROACH =========================== '''
	def add_weights( self , graph ) :
		''' START POINTER FUNCTIONS '''
		bic_score = self.model.bic_score
		''' END POINTER FUNCTIONS '''
		G = self.clean_graph()
		for field in self.data.fields :
			for par in graph[ field ][ 'parents' ] :
				best_parents = copy( graph[ field ][ 'parents' ] )
				new_parents = copy( best_parents )
				new_parents.remove( par )
				weight = bic_score( field , best_parents ) - bic_score( field , new_parents )
				G[ field ][ 'parents' ].append( ( par , weight ) )
				G[ par ][ 'childs' ].append( ( field , weight ) )
		return G
	
	def delete_weights( self , graph ) :
		G = self.clean_graph()
		for field in self.data.fields :
			for ( child , weight ) in graph[ field ][ 'childs' ] :
				G[ field ][ 'childs' ].append( child )
				G[ child ][ 'parents' ].append( field )
		return G
	
	def get_edges( self , graph , cycle ) :
		cycle = list( reversed( cycle ) )
		edges = []
		length = len( cycle )
		''' START POINTER FUNCTIONS '''
		append = edges.append
		''' END POINTER FUNCTIONS '''
		for i in xrange( length ) :
			from_node = cycle[ i ]
			to_node = cycle[ ( i + 1 ) % length ]
			for ( child , weight ) in graph[ from_node ][ 'childs' ] :
				if child == to_node :
					append( ( from_node , to_node , weight ) )
		return edges

	def has_cycles( self , graph ) :
		''' START POINTER FUNCTIONS '''
		index = self.data.fields.index
		lstfields = self.data.fields
		''' END POINTER FUNCTIONS '''
		length = len( lstfields )
		row = [ INT_MAX ] * length
		g = []
		for i in xrange( length ) : g.append( copy( row ) )
		for field in lstfields :
			idx = index( field )
			for ( child , weight ) in graph[ field ][ 'childs' ] :
				idy = index( child )
				g[ idx ][ idy ] = 1
		p = copy( g )
		for i in xrange( length ) :
			for j in xrange( length ) :
				p[ i ][ j ] = i
		for k in xrange( length ) :
			for i in xrange( length ) :
				for j in xrange( length ) :
					aux = g[ i ][ k ] + g[ k ][ j ]
					if aux < g[ i ][ j ] :
						g[ i ][ j ] = aux
						p[ i ][ j ] = p[ k ][ j ]
		cycle = []
		for i in xrange( length ) :
			if g[ i ][ i ] != INT_MAX :
				cycle.append( lstfields[ p[ i ][ i ] ] )
				s = i
				t = p[ i ][ i ]
				while s != t :
					cycle.append( lstfields[ p[ s ][ t ] ] )
					t = p[ s ][ t ]
				return self.get_edges( graph , cycle )
		return None

	def fas_solver( self , graph ) :
		fas_set = []
		while True :
			cycle = self.has_cycles( graph )
			if not cycle : break
			worst_weight = min( [ edg[ 2 ] for edg in cycle ] ) # Tuples ( From , To , Weight )
			for edg in cycle :
				for ( child , weight ) in graph[ edg[ 0 ] ][ 'childs' ] :
					if child != edg[ 1 ] : continue
					graph[ edg[ 0 ] ][ 'childs' ].remove( ( child , weight ) )
					new_weight = weight - worst_weight
					if new_weight == 0 :
						fas_set.append( ( edg[ 0 ] , edg[ 1 ] ) )
					else :
						graph[ edg[ 0 ] ][ 'childs' ].append( ( child , weight - worst_weight ) )
		return self.delete_weights( graph )

	def weighted_solutions( self ) :
		print "Building graph with best parents for each field"
		greedy_graph = self.find_greedy_network( self.data.fields , all_options = True )
		print "GREEDY GRAPH"
		for f in self.data.fields : print "%s:%s" % ( f , greedy_graph[ f ][ 'parents' ] )
		weighted_graph = self.add_weights( greedy_graph )
		fas_graph = self.fas_solver( weighted_graph )
		solutions = []
		''' START POINTER FUNCTIONS '''
		append = solutions.append
		lstfields = self.data.fields
		''' END POINTER FUNCTIONS '''
		for i in xrange( NUM_INITIAL_SOLUTIONS ) :
			print " === Building order #%s === " % (i+1)
			order = topological( fas_graph , lstfields )
			if order in solutions : continue
			append( copy( order ) )
		return solutions

if __name__ == "__main__" :
	if len( sys.argv ) == 3 :
		dataset_file , ommit_fields = sys.argv[ 1: ]
		if ommit_fields == 'None' : ommit_fields = []
		else : ommit_fields = [ f.strip() for f in ommit_fields.split( ',' ) ]
		builder = BNBuilder( dataset_file , savefilter = True , discretize = True , ommit = ommit_fields )
		out_file = "%s%s_%%s.txt" % ( RESULTS_DIR , builder.dataset_name )

		print "========== RUNNING WITH RANDOM PERMUTATION =========="
		builder.setInitialSolutionType( 'random' )
		builder.buildNetwork( outfilepath = out_file % 'random' )
	
		print "========== RUNNING WITH DFS =========="
		builder.setInitialSolutionType( 'unweighted' )
		builder.buildNetwork( outfilepath = out_file % 'unweighted' )
		
		print "========== RUNNING WITH FAS APPROXIMATION =========="
		builder.setInitialSolutionType( 'weighted' )
		builder.buildNetwork( outfilepath = out_file % 'weighted' )
	else :
		print "Usage: pypy %s <csv_file> <ommit_fields> <results_file>" % sys.argv[ 0 ]
