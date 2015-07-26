class UnionFind :
	def __init__( self , size ) :
		self.p = list( xrange( size ) )
	
	def findSet( self , node ) :
		if node == self.p[ node ] : return node
		self.p[ node ] = self.findSet( self.p[ node ] )
		return self.p[ node ]
	
	def sameSet( self , node1 , node2 ) :
		return self.findSet( node1 ) == self.findSet( node2 )

	def unionSet( self , node1 , node2 ) :
		self.p[ self.findSet( node2 ) ] = self.findSet( node1 )
