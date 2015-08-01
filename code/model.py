from copy import deepcopy as copy
from utils import *
from data import Data
from math import log

class Model :
	def __init__( self , dataobj = None , modelfile = None ) :
		if dataobj : self.data = dataobj
		if modelfile : self.loadmodel( modelfile )
	
	def initdict( self ) :
		self.entropyvalues = dict( [ ( field , {} ) for field in self.data.fields ] )
		self.sizevalues = dict( [ ( field , {} ) for field in self.data.fields ] )
		self.bicvalues = dict( [ ( field , {} ) for field in self.data.fields ] )
	
	def loaddata( self , data ) :
		self.data = copy( data )
	
	def addtrainingset( self , testfile , ommit = [] ) :
		print "Adding test rows from %s" % testfile
		testdata = Data( testfile , ommit )
		self.testdata = testdata.rows
		print "TEST ROWS = %s" % len( self.testdata )
	
	def loadmodel( self , modelfile ) :
		self.modelfile = modelfile
		print "Loading model from %s" % modelfile
		fieldset = self.data.fields
		node = { 'parents' : [] , 'childs' : [] }
		self.network = dict( [ ( field , copy( node ) ) for field in fieldset ] )
		with open( modelfile , 'r' ) as f :
			lines = f.readlines()
			for l in lines :
				sp = l[ :-1 ].split( ':' )
				field = sp[ 0 ]
				childs = [ s.strip() for s in sp[ 1 ].split( ',' ) if len( s.strip() ) > 0 ]
				for ch in childs :
					self.network[ field ][ 'childs' ].append( ch )
					self.network[ ch ][ 'parents' ].append( field )
		print "Finding topological order for network"
		self.topological = topological( self.network , fieldset )
		print "Top. Order = %s" % self.topological
	
	def setnetwork( self , network , topo_order = None , train = True ) :
		self.network = copy( network )
		if not topo_order : self.topological = topological( self.network , self.data.fields )
		else : self.topological = topo_order
		if train : self.trainmodel()
	
	def trainmodel( self ) :
		print "Training model..."
		self.probs = dict( [ ( field , {} ) for field in self.data.fields ] )
		for field in self.data.fields :
			xi = [ field ]
			pa_xi = self.network[ field ][ 'parents' ]
			self.calculateprobabilities( xi , pa_xi )
	
	# DATA LOG_LIKELIHOOD
	def testmodel( self ) :
		print "Testing model with test data"
		loglikelihood = 0.0
		for row in self.testdata :
			for field in self.data.fields :
				xi = { field : row[ field ] }
				pa_xi = dict( [ ( pai , row[ pai ] ) for pai in self.network[ field ][ 'parents' ] ] )
				prob = self.conditional_prob( xi , pa_xi )
				loglikelihood += log( prob )
		print "Data Log-Likelihood = %s" % loglikelihood
		return loglikelihood
	
	def loadAndTestModel( self , modelfile ) :
		self.loadmodel( modelfile )
		self.trainmodel()
		return self.testmodel()

	def calculateprobabilities( self , xsetfield , ysetfield ) :
		#print "Calculating P( %s | %s )" % ( xsetfield , ysetfield )
		implies = self.data.evaluate( xsetfield )
		condition = self.data.evaluate( ysetfield )
		for xdict in implies :
			xkey , xval = xdict.keys()[ 0 ] , xdict.values()[ 0 ]
			if xval not in self.probs[ xkey ] : self.probs[ xkey ][ xval ] = {}
			if not condition :
				self.conditional_prob( xdict , {} )
				continue
			for y in condition :
				self.conditional_prob( xdict , y )
	
	def conditional_prob( self , x , y ) :
		xkey , xval = x.keys()[ 0 ] , x.values()[ 0 ]
		cond = self.data.hashed( y )
		if cond in self.probs[ xkey ][ xval ] : return self.probs[ xkey ][ xval ][ cond ]
		numerator = copy( x )
		for key in y : numerator[ key ] = y[ key ]
		denominator = y
		pnum = self.data.getcount( numerator )
		pden = len( self.data.rows ) if not denominator else self.data.getcount( denominator )
		pnum , pden = ( pnum + self.bdeuprior( numerator ) , pden + self.bdeuprior( denominator ) )
		resp = float( pnum ) / float( pden )
		self.probs[ xkey ][ xval ][ cond ] = resp
		return resp
	
	def bdeuprior( self , setfields ) :
		prior = 1.0
		fieldtypes = self.data.fieldtypes
		for field in setfields :
			tam = ( len( self.data.stats[ field ] ) if fieldtypes[ field ] == LITERAL_FIELD else 2 )
			prior *= tam
		return ESS / prior
	
	def score( self ) :
		resp = 0.0
		for field in self.data.fields :
			resp += self.bic_score( field , self.network[ field ][ 'parents' ] )
		self.network[ 'score' ] = resp
		return resp
	
	def bic_score( self , xsetfield , ysetfield ) :
		field = xsetfield
		cond = self.hashedarray( ysetfield )
		if cond in self.bicvalues[ field ] : return self.bicvalues[ field ][ cond ]
		#print "Calculating BIC( %s | %s )" % ( xsetfield , ysetfield )
		N = len( self.data.rows )
		H = self.entropy( xsetfield , ysetfield )
		S = self.size( xsetfield , ysetfield )
		resp = ( -N * H ) + ( log( N ) / 2.0 * S )
		return -resp
	
	def entropy( self , xsetfield , ysetfield ) :
		field = xsetfield
		cond = self.hashedarray( ysetfield )
		if cond in self.entropyvalues[ field ] : return self.entropyvalues[ field ][ cond ]
		x = self.data.evaluate( [ xsetfield ] )
		y = self.data.evaluate( ysetfield )
		N = len( self.data.rows )
		resp = 0.0
		for xdict in x :
			xkey , xval = xdict.keys()[ 0 ] , xdict.values()[ 0 ]
			for ydict in y :
				ij = copy( ydict )
				ijk = copy( ij )
				ijk[ xkey ] = xval
				Nijk = self.data.getcount( ijk ) + self.bdeuprior( ijk )
				Nij = self.data.getcount( ij ) + self.bdeuprior( ij )
				resp += ( Nijk / N * log( Nijk / Nij ) )
		self.entropyvalues[ field ][ cond ] = -resp
		return -resp

	def size( self , xsetfield , ysetfield ) :
		field = xsetfield
		cond = self.hashedarray( ysetfield )
		if cond in self.sizevalues[ field ] : return self.sizevalues[ field ][ cond ]
		resp = len( self.data.evaluate( [ xsetfield ] ) ) - 1
		for field in ysetfield :
			resp *= len( self.data.evaluate( [ field ] ) )
		self.sizevalues[ field ][ cond ] = resp
		return resp
	
	def hashedarray( self , setfields ) :
		resp = ''
		if not setfields : return resp
		for field in self.data.fields :
			if field not in setfields : continue
			resp += "%s, " % field
		return resp[ :-2 ]
