from copy import deepcopy as copy
from utils import *
from data import Data
from math import log

class Model :
	def __init__( self , dataobj = None , modelfile = None ) :
		if dataobj :
			self.data = dataobj
			self.initialize()
		if modelfile :
			self.loadmodel( modelfile )

	def initialize( self ) :
		self.entropyvalues = dict( [ ( field , {} ) for field in self.data.fields ] )
		self.sizevalues = dict( [ ( field , {} ) for field in self.data.fields ] )
		self.bicvalues = dict( [ ( field , {} ) for field in self.data.fields ] )
		self.bestparents = dict( [ ( field , [] ) for field in self.data.fields ] )
		self.precalculate_scores()
	
	def precalculate_scores( self ) :
		score_file = "%s/%s%s" % ( os.path.dirname( self.data.source ) , os.path.splitext( os.path.basename( self.data.source ) )[ 0 ] , '_scores.txt' )
		if os.path.isfile( score_file ) :
			print "Reading from %s all scores" % score_file
			with open( score_file , 'r' ) as f :
				for line in f :
					field , par , sc = line.split()
					if par == '_' : par = ''
					self.bicvalues[ field ][ par ] = float( sc )
					self.bestparents[ field ].append( par.split( ',' ) )
		else :
			print "Pre-calculating all scores from model"
			self.subconj = []
			for i in xrange( 0 , MAX_NUM_PARENTS + 1 ) :
				self.subconj.extend( [ list( x ) for x in itertools.combinations( self.data.fields , i ) ] )
			for field in self.data.fields :
				for sub in self.subconj :
					if field in sub : continue
					self.bic_score( field , sub )
			with open( score_file , 'w' ) as f :
				for field in self.data.fields :
					lstparents = [ ( self.bicvalues[ field ][ p ] , p ) for p in self.bicvalues[ field ] ]
					lstparents.sort( reverse = True )
					''' START POINTER FUNCTION '''
					append = self.bestparents[ field ].append
					''' END POINTER FUNCTION '''
					for ( sc , p ) in lstparents :
						par = copy( p )
						if not par : par = '_'
						f.write( "%s %s %s\n" % ( field , par , self.bicvalues[ field ][ p ] ) )
						append( p )

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
		#print "Training model..."
		''' START POINTER FUNCTIONS '''
		calc_probs = self.calculateprobabilities
		lstfields = self.data.fields
		''' END POINTER FUNCTIONS '''
		self.probs = dict( [ ( field , {} ) for field in lstfields ] )
		for field in self.data.fields :
			xi = [ field ]
			pa_xi = self.network[ field ][ 'parents' ]
			calc_probs( xi , pa_xi )

	# DATA LOG_LIKELIHOOD
	def testmodel( self ) :
		print "Testing model with test data"
		loglikelihood = 0.0
		''' START POINTER FUNCTIONS '''
		lstfields = self.data.fields
		cond_prob = self.conditional_prob
		''' END POINTER FUNCTIONS '''
		for row in self.testdata :
			for field in lstfields :
				xi = { field : row[ field ] }
				pa_xi = dict( [ ( pai , row[ pai ] ) for pai in self.network[ field ][ 'parents' ] ] )
				prob = cond_prob( xi , pa_xi )
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
		resp = ( -N * H ) - ( log( N ) / 2.0 * S )
		#print "BIC( %s | %s ) = %s" % ( xsetfield , ysetfield , resp )
		self.bicvalues[ field ][ cond ] = resp
		return resp

	def entropy( self , xsetfield , ysetfield ) :
		field = xsetfield
		cond = self.hashedarray( ysetfield )
		if cond in self.entropyvalues[ field ] : return self.entropyvalues[ field ][ cond ]
		x = self.data.evaluate( [ xsetfield ] )
		y = self.data.evaluate( ysetfield )
		N = len( self.data.rows )
		resp = 0.0
		''' START POINTER FUNCTIONS '''
		getcount = self.data.getcount
		bdeuprior = self.bdeuprior
		''' END POINTER FUNCTIONS '''
		for xdict in x :
			xkey , xval = xdict.keys()[ 0 ] , xdict.values()[ 0 ]
			if not y :
				Nij = getcount( xdict ) + bdeuprior( xdict )
				resp += ( Nij / N ) * log( Nij / N )
				continue
			for ydict in y :
				ij = copy( ydict )
				ijk = copy( ij )
				ijk[ xkey ] = xval
				Nijk = getcount( ijk ) + bdeuprior( ijk )
				Nij = getcount( ij ) + bdeuprior( ij )
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
		setfields.sort()
		return ','.join( setfields )
