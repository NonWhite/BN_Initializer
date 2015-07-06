from utils import *
from copy import deepcopy as copy
import os

class Extractor :
	def __init__( self , sources , savefilter = False , ommit = [] , discretize = True , fullout = 'out.csv' ) :
		self.sources = sources
		self.savefiltered = savefilter
		self.fields = None
		self.rows = []
		self.ommitedfields = ommit
		self.discretize = discretize
		self.outfile = DATA_DIR + fullout
		self.init()
	
	def init( self ) :
		self.preprocess()
		self.calculatestats()
		self.discretizefields()
		self.export()

	def preprocess( self ) :
		for source in self.sources :
			print "Pre-processing %s" % source
			ext = os.path.splitext( source )[ 1 ]
			outfile = ( source.replace( ext , '_new' + ext ) if self.savefiltered else None )
			out = ( open( outfile , 'w' ) if self.savefiltered else None )
			with open( source , 'r' ) as f :
				lines = f.readlines()
				fields = self.extractFromLine( lines[ 0 ] )
				ommit_positions = [ fields.index( ommit ) for ommit in self.ommitedfields ]
				for v in ommit_positions : fields.remove( fields[ v ] )
				lines = lines[ 1: ]
				print "TOTAL ROWS = %7s" % len( lines )
				if self.fields and self.fields != fields :
					print "SOURCES DOESN'T HAVE SAME FIELDS"
					return
				self.fields = fields
				newrows = [ self.extractFromLine( l , ommit_positions ) for l in lines if l.find( '?' ) < 0 ]
				self.rows.extend( newrows )
				print "REMOVED ROWS = %5s" % ( len( lines ) - len( newrows ) )
				if self.savefiltered :
					out.write( ','.join( self.fields ) + '\n' )
					for row in newrows :
						out.write( ','.join( row ) + '\n' )
				print "FILTER ROWS = %6s" % len( newrows )
			if self.savefiltered :
				print "Created %s!!!" % outfile
		self.rowsToDict()
		self.analyzeFields()
	
	def rowsToDict( self ) :
		dictrows = []
		for row in self.rows :
			newrow = {}
			for i in range( len( self.fields ) ) :
				field = self.fields[ i ]
				value = row[ i ]
				newrow[ field ] = value
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
					if value not in self.stats[ field ] : self.stats[ field ][ value ] = 0
					self.stats[ field ][ value ] += 1
				else :
					if not self.stats[ field ] : self.stats[ field ] = copy( num_stats )
					value = int( value )
					self.stats[ field ][ 'min' ] = min( self.stats[ field ][ 'min' ] , value )
					self.stats[ field ][ 'max' ] = max( self.stats[ field ][ 'max' ] , value )
					self.stats[ field ][ 'mean' ] += int( value )
					self.stats[ field ][ 'median' ].append( value )
		for field in self.fields :
			if self.fieldtypes[ field ] == NUMERIC_FIELD :
				self.stats[ field ][ 'mean' ] /= len( self.rows )
				length = len( self.stats[ field ][ 'median' ] )
				self.stats[ field ][ 'median' ] = sorted( self.stats[ field ][ 'median' ] )[ length / 2 ]
	
	def discretizefields( self ) :
		if not self.discretize : return
		for field in self.fields :
			if self.fieldtypes[ field ] != NUMERIC_FIELD : continue
			for row in self.rows :
				row[ field ] = ( 1 if int( row[ field ] ) > self.stats[ field ][ 'median' ] else 0 )
	
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
			print " DIFERENT VALUES = %s" % len( self.stats[ field ].keys() )
			if self.fieldtypes[ field ] == LITERAL_FIELD :
				values = [ ( count , val ) for ( val , count ) in self.stats[ field ].iteritems() ]
				values = sorted( values , reverse = True )
				for val in values : print "%s: %s" % ( val[ 1 ] , val[ 0 ] )
			else :
				print "Min = %s" % self.stats[ field ][ 'min' ]
				print "Max = %s" % self.stats[ field ][ 'max' ]
				print "Mean = %s" % self.stats[ field ][ 'mean' ]
				print "Median = %s" % self.stats[ field ][ 'median' ]
	
if __name__ == "__main__" :
	sources = [ TRAINING_FILE , TEST_FILE ]
	extractor = Extractor( sources , savefilter = True , ommit = [ 'fnlgwt' ] , discretize = True )
	extractor.printstats()
