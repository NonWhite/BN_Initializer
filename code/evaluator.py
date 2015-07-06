from utils import *
from copy import deepcopy as copy
from math import log
from extractor import Extractor
import sys
import os.path
from random import randint as random

class Evaluator( Extractor ) :
	def addTrainingSet( self , testfile ) :
		testdata = Extractor( testfile , ommit = self.ommitedfields )
		print "Adding test rows from %s" % testfile
		try :
			self.testdata.extend( testdata.rows )
		except :
			self.testdata = []
			self.testdata.extend( testdata.rows )
		print "TEST ROWS = %s" % len( self.testdata )
	
	def loadmodel( self , modelfile ) :
		self.modelfile = modelfile
		print "Loading model from %s" % modelfile
		node = { 'parents' : [] , 'childs' : [] }
		self.network = dict( [ ( field , copy( node ) ) for field in self.fields ] )
		with open( modelfile , 'r' ) as f :
			lines = f.readlines()
			for l in lines :
				sp = l[ :-1 ].split( ':' )
				field = sp[ 0 ]
				childs = [ s.strip() for s in sp[ 1 ].split( ',' ) if len( s.strip() ) > 0 ]
				for ch in childs :
					self.network[ field ][ 'childs' ].append( ch )
					self.network[ ch ][ 'parents' ].append( field )
		self.topological = topological( self.network , self.fields )
		#for field in self.fields :
		#	print "%s: %s" % ( field , self.network[ field ][ 'childs' ] )

	def trainmodel( self ) :
		self.probs = dict( [ ( field , {} ) for field in self.fields ] )
		for field in self.network :
			xi = [ field ]
			pa_xi = [ f for f in self.network[ field ][ 'parents' ] ]
			self.calculateprobabilities( xi , pa_xi )

	# DATA LOG_LIKELIHODD
	def testmodel( self ) :
		print "Testing model with test data"
		loglikelihood = 0.0
		for row in self.testdata :
			for field in self.fields :
				xi = { field: row[ field ] }
				pa_xi = dict( [ ( pai , row[ pai ] ) for pai in self.network[ field ][ 'parents' ] ] )
				prob = self.conditional_prob( xi , pa_xi )
				loglikelihood += log( prob )
		print "Data Log-Likelihood = %s" % loglikelihood
		return loglikelihood
	
	def calculateprobabilities( self , xsetfield , ysetfield ) :
		print "Calculating P( %s | %s )" % ( xsetfield , ysetfield )
		implies = self.evaluate( xsetfield )
		condition = self.evaluate( ysetfield )
		self.cont = 0
		for xdict in implies :
			xkey = xdict.keys()[ 0 ] # Supose it only has one field in X
			xval = xdict.values()[ 0 ]
			if xval not in self.probs[ xkey ] : self.probs[ xkey ][ xval ] = {}
			if not condition :
				self.conditional_prob( xdict , {} )
				continue
			for y in condition :
				self.conditional_prob( xdict , y )
		print "CONT = %s" % self.cont

	def conditional_prob( self , x , y ) :
		xkey , xval = x.keys()[ 0 ] , x.values()[ 0 ]
		cond = self.hashed( y )
		if cond in self.probs[ xkey ][ xval ] : return self.probs[ xkey ][ xval ][ cond ]
		self.cont += 1
		if self.cont % 1000 == 0 : print "CONT = %s" % self.cont
		numerator = copy( x )
		for key in y : numerator[ key ] = y[ key ]
		denominator = y
		if not denominator :
			pnum = len( self.query( numerator ) )
			pden = len( self.rows )
		else :
			pnum = len( self.query( numerator ) ) + self.bdeuprior( numerator )
			pden = len( self.query( denominator ) ) + self.bdeuprior( denominator )
		resp = float( pnum ) / float( pden )
		self.probs[ xkey ][ xval ][ cond ] = resp
		return resp

	def hashed( self , cond ) :
		resp = ''
		if not cond : return resp
		for field in self.fields :
			if field not in cond : continue
			resp += "%s:%s, " % ( field , cond[ field ] )
		return resp[ :-2 ]
	
	def hashedarray( self , setfields ) :
		resp = ''
		if not setfields : return resp
		for field in self.fields :
			if field not in setfields : continue
			resp += "%s, " % field
		return resp[ :-2 ]
	
	def query( self , filters ) :
		resp = []
		for row in self.rows :
			if all( item in row.items() for item in filters.items() ) :
				resp.append( row )
		return resp
	
	def bdeuprior( self , setfields ) :
		prior = 1.0
		for field in setfields :
			tam = ( len( self.stats[ field ] ) if self.fieldtypes[ field ] == LITERAL_FIELD else 2 )
			prior *= tam
		return ESS / prior

	def evaluate( self , setfields , pos = 0 ) :
		if pos == len( setfields ) : return []
		field = setfields[ pos ]
		if self.fieldtypes[ field ] == NUMERIC_FIELD :
			values = [ 0 , 1 ]
		else :
			values = self.stats[ field ].keys()
		resp = []
		for x in values :
			node = { field: x }
			nxt = self.evaluate( setfields , pos + 1 )
			if not nxt :
				resp.append( copy( node ) )
				continue
			for r in nxt :
				r[ field ] = x
				resp.append( r )
		return resp
	
	def loadAndTestModel( self , modelfile ) :
		self.loadmodel( modelfile )
		self.trainmodel()
		return self.testmodel()
	
	def synthethicData( self , modelname ) :
		modelname = os.path.basename( modelname )
		rows_to_generate = int( GENERATED_DATA * TRAINING_DATA_PERCENTAGE )
		self.generateData( GEN_TRAINING_FILE % modelname , rows_to_generate )
		rows_to_generate = int( GENERATED_DATA * TEST_DATA_PERCENTAGE )
		self.generateData( GEN_TEST_FILE % modelname , rows_to_generate )
	
	def generateData( self , filename , num_rows ) :
		print "Generating data (%s rows) in %s" % ( num_rows , filename )
		with open( filename , 'w' ) as f :
			f.write( ','.join( self.topological ) + '\n' )
			for x in range( num_rows ) :
				row = self.generateRow()
				line = [ row[ field ] for field in self.topological ]
				f.write( ','.join( line ) + '\n' )
	
	def generateRow( self ) :
		row = {}
		for field in self.topological :
			row[ field ] = self.generateValue( field , row )
		return row

	def generateValue( self , field , row ) :
		parents = dict( [ ( f , row[ f ] ) for f in self.network[ field ][ 'parents' ] ] )
		values = self.evaluate( [ field ] )
		probs = []
		for val in values :
			cond_prob = self.conditional_prob( val , parents )
			probs.append( ( val[ field ] , int( SIZE_TO_GET_RAND_VALUE * cond_prob ) ) )
		rand = []
		for ( val , q ) in probs :
			for x in range( q ) :
				rand.append( val )
		return str( shuffle( rand )[ 0 ] )
	
if __name__ == "__main__" :
	training_data = TRAINING_FILE
	test_data = TEST_FILE
	models = MODELS
	if len( sys.argv ) > 1 :
		( training_data , test_data ) = sys.argv[ 1: ]

	evaluator = Evaluator( [ training_data ] , savefilter = True , ommit = [ 'fnlgwt' ] , discretize = True )
	evaluator.addTrainingSet( [ test_data ] )
	for mod in models :
		evaluator.loadAndTestModel( mod )
		evaluator.synthethicData( mod )
