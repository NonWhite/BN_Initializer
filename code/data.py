from utils import *
from copy import deepcopy as copy
import os.path

class Data :
	def __init__( self , source , savefilter = False , ommit = [] , discretize = True , outfile = 'out.log' ) :
		self.source = source
		self.savefiltered = savefilter
		self.fields = None
		self.rows = []
		self.ommitedfields = ommit
		self.discretize = discretize
		self.outfile = RESULTS_DIR + outfile
		self.init()

	def init( self ) :
		self.preprocess()
		self.calculatestats()
		self.printstats()
		self.discretizefields()
		self.export()

	def preprocess( self ) :
		source = self.source
		print "Pre-processing %s" % source
		ext = os.path.splitext( source )[ 1 ]
		outfile = ( source.replace( ext , '_new' + ext ) if self.savefiltered else None )
		out = ( open( outfile , 'w' ) if self.savefiltered else None )
		with open( source , 'r' ) as f :
			lines = f.readlines()
			self.fields = self.extractFromLine( lines[ 0 ] )
			ommit_pos = [ self.fields.index( ommit ) for ommit in self.ommitedfields ]
			for v in ommit_pos : self.fields.remove( self.fields[ v ] )
			self.fields = [ field.replace( '-' , '_' ) for field in self.fields ]
			print self.fields
			lines = lines[ 1: ]
			print "TOTAL ROWS = %7s" % len( lines )
			newrows = [ self.extractFromLine( l , ommit_pos ) for l in lines if l.find( '?' ) < 0 ]
			self.rows = newrows
			print "REMOVED ROWS = %5s" % ( len( lines ) - len( newrows ) )
			if self.savefiltered :
				out.write( ','.join( self.fields ) + '\n' )
				for row in newrows : out.write( ','.join( row ) + '\n' )
			print "FILTER ROWS = %6s" % len( newrows )
		if self.savefiltered : print "Created %s!!!" % outfile
		self.rowsToDict()
		self.analyzeFields()

	def rowsToDict( self ) :
		dictrows = []
		for row in self.rows :
			newrow = dict( [ ( field , row[ self.fields.index( field ) ] ) for field in self.fields ] )
			dictrows.append( newrow )
		self.rows = dictrows

	def analyzeFields( self ) :
		self.fieldtypes = dict( [ ( field , '' ) for field in self.fields ] )
		for field in self.fields :
			if self.rows[ 0 ][ field ].isdigit() :
				self.fieldtypes[ field ] = NUMERIC_FIELD
			else :
				self.fieldtypes[ field ] = LITERAL_FIELD

	def extractFromLine( self , line , ommit_positions = [] ) :
		x = line[ :-1 ].split( FIELD_DELIMITER )
		x = [ v.strip() for v in x ]
		for v in ommit_positions : x.remove( x[ v ] )
		return x

	def calculatestats( self ) :
		self.stats = dict( [ ( field , {} ) for field in self.fields ] )
		num_stats = { 'min' : INT_MAX , 'max' : -INT_MAX , 'mean' : 0.0 , 'median' : [] }
		for row in self.rows :
			for field in self.fields :
				value = row[ field ]
				if self.fieldtypes[ field ] == LITERAL_FIELD :
					self.stats[ field ][ value ] = self.stats[ field ].get( value , 0 ) + 1
				else :
					if not self.stats[ field ] : self.stats[ field ] = copy( num_stats )
					value = float( value )
					self.stats[ field ][ 'min' ] = min( self.stats[ field ][ 'min' ] , value )
					self.stats[ field ][ 'max' ] = max( self.stats[ field ][ 'max' ] , value )
					self.stats[ field ][ 'mean' ] += value
					self.stats[ field ][ 'median' ].append( value )
		for field in self.fields :
			if self.fieldtypes[ field ] == NUMERIC_FIELD :
				self.stats[ field ][ 'mean' ] /= len( self.rows )
				le = len( self.stats[ field ][ 'median' ] )
				self.stats[ field ][ 'median' ] = sorted( self.stats[ field ][ 'median' ] )[ le / 2 ]

	def discretizefields( self ) :
		if not self.discretize : return
		for field in self.fields :
			if self.fieldtypes[ field ] != NUMERIC_FIELD : continue
			for row in self.rows :
				row[ field ] = ( 1 if row[ field ] > self.stats[ field ][ 'median' ] else 0 )

	def calculatecounters( self ) :
		print "Pre-calculating all queries from data"
		self.counters = {}
		self.subconj = []
		for i in range( 1 , len( self.fields ) + 1 ) :
			self.subconj.extend( [ list( x ) for x in itertools.combinations( self.fields , i ) ] )
		self.rows = self.rows[ :100 ] # TODO: Delete
		for idx in range( len( self.rows ) ) :
			row = self.rows[ idx ]
			for sub in self.subconj :
				H = self.hashed( getsubconj( row , sub ) )
				if H not in self.counters : self.counters[ H ] = 0.0
				self.counters[ H ] += 1.0
			if idx % 1000 == 0 : print idx

	def getcount( self , fields ) :
		F = self.hashed( fields )
		if F not in self.counters : self.counters[ F ] = 0.0
		return self.counters[ F ]

	def hashed( self , cond ) :
		resp = ''
		if not cond : return resp
		for field in self.fields :
			if field not in cond : continue
			resp += "%s:%s, " % ( field , cond[ field ] )
		return resp[ :-2 ]

	def export( self ) :
		if not self.savefiltered : return
		with open( self.outfile , 'w' ) as f :
			f.write( ','.join( self.fields ) + '\n' )
			for row in self.rows :
				line = ','.join( [ str( row[ field ] ) for field in self.fields ] )
				f.write( line + '\n' )

	def printstats( self ) :
		print "TOTAL ENTITIES = %s" % len( self.rows )
		for field in self.stats :
			print " ======== FIELD: %s ======== " % field
			diffvalues = len( self.stats[ field ].keys() )
			if self.fieldtypes[ field ] == NUMERIC_FIELD : diffvalues = 2
			print " DIFFERENT VALUES = %s" % diffvalues
			if self.fieldtypes[ field ] == LITERAL_FIELD :
				values = [ ( count + 0.0 , val ) for ( val , count ) in self.stats[ field ].iteritems() ]
				values = sorted( values , reverse = True )
				for val in values :
					percentage = val[ 0 ] / len( self.rows ) * 100.0
					print "%-30s: %6s (%5.2f %%)" % ( val[ 1 ] , int( val[ 0 ] ) , percentage )
			else :
				print "Min = %s" % self.stats[ field ][ 'min' ]
				print "Max = %s" % self.stats[ field ][ 'max' ]
				print "Mean = %s" % self.stats[ field ][ 'mean' ]
				print "Median = %s" % self.stats[ field ][ 'median' ]

	def evaluate( self , setfields , pos = 0 ) :
		if pos == len( setfields ) : return []
		field = setfields[ pos ]
		fieldtype = self.fieldtypes[ field ]
		if fieldtype == NUMERIC_FIELD :
			values = [ 0 , 1 ]
		else :
			values = self.stats[ field ].keys()
		resp = []
		for x in values :
			node = { field : x }
			nxt = self.evaluate( setfields , pos + 1 )
			if not nxt :
				resp.append( copy( node ) )
				continue
			for r in nxt :
				r[ field ] = x
				resp.append( r )
		return resp
