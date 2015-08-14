from utils import *
from copy import deepcopy as copy
import os.path
import sys

class Data :
	def __init__( self , source , savefilter = False , ommit = [] , discretize = True , outfile = 'out.csv' ) :
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
				if compare( self.stats[ field ][ 'max' ] - self.stats[ field ][ 'min' ] , 1.0 ) == 0 :
					continue
				if compare( float( row[ field ] ) , float( self.stats[ field ][ 'median' ] ) ) > 0 :
					row[ field ] = 1
				else :
					row[ field ] = 0

	def calculatecounters( self ) :
		counter_file = "%s/%s%s" % ( os.path.dirname( self.source ) , os.path.splitext( os.path.basename( self.source ) )[ 0 ] , '_counters.txt' )
		self.counters = {}
		if os.path.isfile( counter_file ) :
			print "Reading from %s all counters" % counter_file
			with open( counter_file , 'r' ) as f :
				lines = [ l[ :-1 ].split( ' ' ) for l in f.readlines() ]
				for L in lines :
					self.counters[ L[ 0 ] ] = float( L[ 1 ] )
		else :
			print "Pre-calculating all queries from data"
			self.subconj = []
			for i in xrange( 1 , MAX_NUM_PARENTS + 2 ) :
				self.subconj.extend( [ list( x ) for x in itertools.combinations( self.fields , i ) ] )
			for idx in xrange( len( self.rows ) ) :
				row = self.rows[ idx ]
				for sub in self.subconj :
					H = self.hashed( getsubconj( row , sub ) )
					if H not in self.counters : self.counters[ H ] = 0.0
					self.counters[ H ] += 1.0
				if idx % 1000 == 0 : print idx
			print "Saving in %s all counters" % counter_file
			with open( counter_file , 'w' ) as f :
				for ( key , value ) in self.counters.iteritems() :
					f.write( "%s %s\n" % ( key , value ) )

	def getcount( self , fields ) :
		F = self.hashed( fields )
		if F not in self.counters : self.counters[ F ] = 0.0
		return self.counters[ F ]

	def hashed( self , cond ) :
		resp = ''
		if not cond : return resp
		for field in self.fields :
			if field not in cond : continue
			resp += "%s:%s," % ( field , cond[ field ] )
		return resp[ :-1 ]

	def export( self ) :
		if not self.savefiltered : return
		with open( self.outfile , 'w' ) as f :
			f.write( ','.join( self.fields ) + '\n' )
			for row in self.rows :
				line = ','.join( [ str( row[ field ] ) for field in self.fields ] )
				f.write( line + '\n' )

	def printstats( self ) :
		print "TOTAL ENTITIES = %s" % len( self.rows )
		print "TOTAL FIELDS = %s" % len( self.fields )
		avg_vals_per_var = 0.0
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
			avg_vals_per_var += diffvalues
		print "AVG #VALS/VAR = %s" % ( avg_vals_per_var / len( self.fields ) )

	def evaluate( self , setfields , pos = 0 ) :
		if pos == len( setfields ) : return []
		field = setfields[ pos ]
		fieldtype = self.fieldtypes[ field ]
		if fieldtype == NUMERIC_FIELD :
			if self.discretize :
				values = [ 0 , 1 ]
			else :
				values = range( int( self.stats[ field ][ 'min' ] ) , int( self.stats[ field ][ 'max' ] ) + 1 )
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

if __name__ == '__main__' :
	if len( sys.argv ) == 2 :
		datasetfile = sys.argv[ 1 ]
		data = Data( datasetfile )
		data.printstats()
