from utils import *
from copy import deepcopy as copy
from data import Data
from model import Model
import os.path
import sys

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
		for k in xrange( NUM_ORDERING_SAMPLES ) :
			lst_fields = shuffle( self.data.fields )
			network = dict( [ ( field , copy( node ) ) for field in self.data.fields ] )
			network[ 'score' ] = 0.0
			self.out.write( "Building network #%s\n" % ( k + 1 ) )
			print "Building network #%s" % ( k + 1 )
			for i in xrange( len( lst_fields ) ) :
				field = lst_fields[ i ]
				print "======== Field #%s: %s ========" % ( i , field )
				best_parents = []
				best_score = ( INT_MAX if i > 0 else self.model.bic_score( field , best_parents ) )
				for t in xrange( NUM_GREEDY_RESTARTS ) :
					if i == 0 : break # First field doesn't have parents
					parents = []
					max_num_parents = min( MAX_NUM_PARENTS , i )
					for n in xrange( max_num_parents ) :
						while True :
							pos = randint( 0 , i - 1 )
							new_parent = lst_fields[ pos ]
							if new_parent not in parents : break
						parents.append( new_parent )
						current = self.model.bic_score( field , parents )
						if compare( current , best_score ) < 0 :
							best_score = current
							best_parents = copy( parents )
				self.addRelation( network , field , best_parents , best_score )
			self.printnetwork( network )
			best_networks.append( copy( network ) )
		sorted( best_networks , key = lambda netw : netw[ 'score' ] , reverse = True )
		return best_networks[ 0 ]

	def greedySearch( self ) :
		self.model.entropyvalues = dict( [ ( field , {} ) for field in self.data.fields ] )
		self.model.sizevalues = dict( [ ( field , {} ) for field in self.data.fields ] )
		self.model.bicvalues = dict( [ ( field , {} ) for field in self.data.fields ] )
		print "Building bayesian network"
		best_model = self.getinitialorder()
		for k in xrange( NUM_GREEDY_ITERATIONS ) :
			print "Iteration #%s" % (k+1)
			cur_model = self.find_order( best_model )
			if compare( cur_model.score() , best_model.score()  ) < 0 :
				best_model = copy( cur_model )
			self.printnetwork( cur_model.network )
		return best_model.network

	def getinitialorder( self ) :
		model = copy( self.model )
		if self.initialrandom :
			order = shuffle( self.data.fields )
			network = self.find_greedy_network( order , all_options = False )
			model.setnetwork( network , topo_order = order )
		else :
			print "Building graph with best parents for each field"
			greedy_graph = self.find_greedy_network( self.data.fields , all_options = True )
			best_score = INT_MAX
			print "Getting feedback arc set for each node"
			print "GREEDY GRAPH"
			for f in self.data.fields : print "%s: %s" % ( f , greedy_graph[ f ][ 'parents' ] )
			for field in self.data.fields :
				print " === Building network with root %s === " % field.upper()
				network = self.clean_graph()
				visited = {}
				graph = copy( greedy_graph )
				self.dfs( graph , field , visited , network )
				print "Changing some edges directions"
				self.changedirections( graph , network )
				#print "NETWORK"
				#for f in self.data.fields : print "%s: %s" % ( f , network[ f ][ 'childs' ] )
				self.model.setnetwork( network )
				print "Calculating BIC Score for network"
				cur_score = self.model.score()
				if compare( cur_score , best_score ) < 0 :
					best_score = cur_score
					model = copy( self.model )
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
			#new_order = copy( order )
			#new_order[ i ] , new_order[ i + 1 ] = new_order[ i + 1 ] , new_order[ i ]
			#network = self.find_greedy_network( new_order )
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
		for p in xrange( MAX_NUM_PARENTS ) :
			opt_parents = []
			opt_score = INT_MAX
			for opt in options :
				cur_parents = copy( best_parents )
				cur_parents.append( opt )
				cur_score = self.model.bic_score( field , cur_parents )
				if compare( cur_score , opt_score ) < 0 :
					opt_score = cur_score
					opt_parents = copy( cur_parents )
			if compare( opt_score , best_score ) < 0 :
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

	def setInitialRandom( self , initialrandom ) :
		self.initialrandom = initialrandom

if __name__ == "__main__" :
	if len( sys.argv ) == 5 :
		training_file , test_file , ommit_fields , out_file = sys.argv[ 1: ]
		ommit_fields = [ f.strip() for f in ommit_fields.split( ',' ) ]
		''' Run with initial random '''
		print "========== RUNNING WITH RANDOM PERMUTATION =========="
		builder = BNBuilder( training_file , savefilter = True , ommit = ommit_fields , initialrandom = True )
		builder.buildNetwork( outfilepath = out_file % 'random' )
		builder.addTrainingSet( test_file )
		loglikelihood = builder.loadAndTestModel( builder.modelfile )
		builder.out.write( "DATA LOG-LIKELIHOOD = %s" % loglikelihood )
		
		''' Run with initial solution '''
		print "========== RUNNING WITH INITIAL SOLUTION =========="
		builder.setInitialRandom( False )
		builder.buildNetwork( outfilepath = out_file % 'heuristic' )
		builder.addTrainingSet( test_file )
		loglikelihood = builder.loadAndTestModel( builder.modelfile )
		builder.out.write( "DATA LOG-LIKELIHOOD = %s" % loglikelihood )
