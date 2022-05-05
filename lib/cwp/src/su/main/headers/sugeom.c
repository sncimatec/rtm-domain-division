/* Copyright (c) Colorado School of Mines, 2011.*/
/* All rights reserved.                       */

/* SUGEOM: $Revision: 1.00 $ ; $Date: 2016/01/04 00:09:00 $        */


#include "sugeom.h"

/*********************** self documentation **********************/
char *sdoc[] = {
"                          ",
" SUGEOM - Fill up geometry in trace headers.                              ",
"                                                                          ",
" sugeom rps= sps= xps=  <stdin >stdout [optional parameters]              ",
"                                                                          ",
" Required parameters:                                                     ",
"       rps=  filename for the receiver points definitions. (AKA R file)   ",
"       sps=  filename for the source   points definitions. (AKA S file)   ",
"       xps=  filename for the relation specification.      (AKA X file)   ",
"                                                                          ",
" OBS:  if 'prefix' (see bellow) is given the last three parameters become ",
"       optional, but any one can be used as an override for the name      ",
"       built using prefix.                                                ",
"          Ex: sugeom prefix=blah  xps=bleh.xxx                            ",
"              Used filenames:  blah.s blah.r and bleh.xxx                 ",
"                                                                          ",
" Optional parameters:                                                     ",
"       prefix=  prefix name for the SPS files.  If given the filename will",
"                be constructed as '<prefix>.s', '<prefix>.r' and          ",
"                '<prefix>.x'.                                             ",
"       rev=0             for SPS revision 0 format.                       ",
"          =2 (Default)   for SPS revision 2.1 format.                     ",
"   verbose=0 (Default)   silent run.                                      ",
"          =1             verbose stats.                                   ",
"     ibin=0  (default)   inline binsize (cdp interval)                    ",
"                         0 -> do not compute cdp number                   ",
"                                                                          ",
"  For SPS format specification consult:                                   ",
"    http://www.seg.org/resources/publications/misc/technical-standards    ",
"                                                                          ",
"                                                                          ",
"  WARNING: This is not a full fledged geometry program.                   ",
"                                                                          ",
"  The SPS format is fully described in two documents available at the     ",
"  above SEG address.                                                      ",
"                                                                          ",
"  To make short what can be a loooong discussion, the SPS file format is  ",
"  intended to describe completely a land (or tranzition zone) survey. The ",
"  main difference between land and marine survey is that in land all      ",
"  points are meant to be previously located and remain fixed along        ",
"  acquisition operation.  In a marine survey the cables and boat can be   ",
"  dragged by current and wind, making the prosition of source and receivers",
"  unpredictable.  The planning and survey description of a marine program ",
"  must take into account all moving points, being necessary a full        ",
"  positioning description of source and receivers at every single shot.   ",
"                                                                          ",
"  The SPS format standard is composed of three text files with 80 columns ",
"  each.  Two of those files are Point Record Specification and are used to",
"  describe all survey points, that is the receiver stations and source    ",
"  points.  The remaining file is the Relation Record Specification and is ",
"  used to describe how each source point is related to a set of           ",
"  corresponding receiver points during the registration of each record    ",
"  (fldr).                                                                 ",
"                                                                          ",
"  These files usually have a number of lines at the start describing the  ",
"  survey and the projections used for coordinates.  This program just     ",
"  skip them.  Those are header entries and have an 'H' in first column.   ",
"                                                                          ",
"  Each line (entry) of the  Point Record Specification contain the        ",
"  information of a single point.  The only difference of a source point   ",
"  specification entry and a receiver point specification entry is the     ",
"  first  column of the file, it has an 'S' for a source point description ",
"  and an 'R' for a receiver point description.  For each point entry there",
"  is an identification with a pair of informations, the Line it belongs   ",
"  and the Point Number, a set of coordinates (X, Y, surface Elevation,    ",
"  depth of element), and the static correction.  All source points        ",
"  description are in a single file (known as S-File), and all receiver    ",
"  station informations are in another file (R-File).                      ",
"                                                                          ",
"  The Relation Record Specification (X-File) is a file with as many       ",
"  entries (text lines) as necessary to completely describe each record    ",
"  (fldr) acquired.   Each entry containing a record information starts    ",
"  with an 'X' in first column.  The informations in the entry starts with ",
"  the tape number in which the register was recorded (not used for this   ",
"  program) and the Field Record Number (fldr).  Next comes the source     ",
"  point description (the same Line and Point(ep) from the S-File). Then a ",
"  sequence of channels (tracf) numbers (just first and last) and a channel",
"  increment.  Next comes the receiver description using the same Line and ",
"  Point/station from the R-File).  The Receiver Points are specified as the",
"  first and last station used at that line.  Finally comes the recording  ",
"  time.   If the spread has a gap, very common for older lines, it will   ",
"  require at least two entries (text lines) to describe this record, one  ",
"  describing the channels and receiver stations before the gap, and another",
"  for the channels and stations after the gap.   The initial informations ",
"  (tape, record, and source) are repeated for all entries.   In a 3D there",
"  is one entry for each line of the patch.                                ",
"                                                                          ",
"  To use this program it is necessary to use all three SPS files.  The    ",
"  S-File and the R-File can describe more points than will be used for the",
"  processing.  For example, if one is processing just a section of a 2D   ",
"  line the Point Record Specification files can describe all points of the",
"  complete line, the information in excess will be disregarded.           ",
"                                                                          ",
"  The X-File must have the information in the same order of the input SU  ",
"  file, if not so all files (fldr) that do not match the order will be    ",
"  skipped until the program find the fldr corresponding to the sequence of",
"  the X-File.                                                             ",
"                                                                          ",
"  Although this program read all of the SPS files, it is not ready for 3D ",
"  geometry processing.   It is just a basic 2D straight line geometry     ",
"  processing program that (hopefully) will fill correctly the informations",
"  in header.  It can be used for a crooked line for the coordinates (X, Y ",
"  Elevation, and element depth), the static correction, and offset.  The  ",
"  cdp numbering will need an specialized program for computation.         ",
"                                                                          ",
"  All coordinates are expected to be an UTM projection in meters or feet. ", 
"                                                                          ",
"  The X file order must be the same as the input file.  Upon reading      ",
"  a trace whose trcf is not the next record to be processed in X file,    ",
"  this record, and the following, will be skipped until a match is found  ",
"  for the current entry in X file.   This is a way to, e.g., skip a noisy ",
"  record, just remove its entry from X file.                              ",
"                                                                          ",
"  If a trace (understand fldr/tracf pair) is not represented in the X file",
"  it will be skipped until a trace in current fldr matches the current    ",
"  set of X entry for that fldr.  If it is desirable to process just a     ",
"  subset of channels, just keep in the X-File information about those     ",
"  channels.                                                               ",
"                                                                          ",
"  Updated trace header positions:                                         ",
"                                                                          ",
"  tracl - it keeps counting to overcome possible trace skipping.          ",
"  ep    - If zeroed out in header it uses the relation (X) file info.     ",
"                                                                          ",
"  sx, sy, gx, gy -  will be updated with the coordinates in point files   ",
"                    R and S. The scalco value used is the already         ",
"                    stored in header, almost ever equal zero.             ",
"  gelev, selev, sdepth - will be updated with values in point files       ",
"                    R and S. The scalel value used is the already         ",
"                    stored in header, almost ever equal zero.             ",
"  sstat, gstatt - these values are filled with the information of the     ",
"           Point Record Specification files.  If not available will be zero.",
"  offset - is computed from source and receiver coordinates.  The offset  ",
"           signal is made negative if source station number is greater    ",
"           than the receiver station number.  It consider that source and ",
"           receiver numbering are the same.   Fractionary stations, e.g.  ",
"           source point halfway between stations are ok.                  ",
"  cdp    - if parameter 'ibin' is passed it is considered the cdp spacing ",
"           and the cdp number will be computed as follow:                 ",
"           The cdp position is computed as the midpoint between source    ",
"           and receiver.  The distance from this point to the first       ",
"           station of R point file is divided by 'ibin' and added to 100. ",
"           This 100 is absolutely artitrary.  ;)                          ",
"                                                                          ",
NULL};

/* Credits:
 * Fernando Roxo <petro@roxo.org>
 *
 * Trace header fields accessed:             
 * fldr, tracf, scalco, scalel, ep, sx, sy, gx, gy, gelev, selev, sdepth, cdp
 */
/**************** end self doc ***********************************/



// Storage area for receiver points
struct PointInfo *RecInfo;

// Storage area for source points
struct PointInfo *SrcInfo;

// Storage area for record informations
struct RegInfo *FileInfo;


int main(int argc, char **argv)
{

   cwp_String prefix=NULL; /* Common prefix to build the SPS filenames. */
   cwp_String Rname=NULL;  /* file name for R file                 */
   int numR=0;             /* number of receiver points            */
   FILE *fpR=NULL;         /* file pointer for R file              */
   cwp_String Sname=NULL;  /* file name for S file                 */
   FILE *fpS=NULL;         /* file pointer for S file              */
   int numS=0;             /* number of source points              */
   cwp_String Xname=NULL;  /* file name for X file                 */
   FILE *fpX=NULL;         /* file pointer for X file              */
   int numX=0;             /* number of relation points entries    */
                           /* WARNING: a record can have more than */
                           /*          one relation entry.  E.g. 3D*/
   int rev;                /* SPS revision to process              */

   int   RC;               /* Return code from functions           */
   int   i;                /* Just the "i".                        */
   float scalco;           /* scale factor                         */
   float scalel;           /* scale factor                         */
   float temp;             /* temporary float value                */

   double Xcdp;            /* CDP X Coordinate                     */
   double Ycdp;            /* CDP Y Coordinate                     */
   float ibin;             /* Inline bin size/cdp interval         */
   //float xbin;             /* Xline bin size/ Not used             */

   float dt;               /* sample spacing - I don't need, just check */
   //int nt;                 /* number of points on input trace      */
   cwp_Bool seismic;       /* is this seismic data?      */

   // File control
   int oldfldr;  //  fldr for last trace processed
   int oldskip;  //  still skipping the same fldr?
   int curx;     //  relation entry control
   int curtrl;   //  keeping track of current trace (in case of skip)

   int idxR;     //  Index of receiver station entry
   int idxS;     //  Index of shot point entry
   float recsta; //  receiver station
   //float srcsta; //  source point station

   int nproct = 0; // number of processed traces
   //int nprocf = 0; // number of processed files
        
   /* Initialize */
   initargs(argc, argv);
   requestdoc(1);

   /* Verbose run? */
   if (!getparint("verbose", &verbose))   verbose=0;

   /* get SPS revision to process */
   if (!getparint("rev", &rev))   rev=2;

   /* get inline bin interval     */
   if (!getparfloat("ibin", &ibin))   ibin=0.;

   /* lets prepare the geometry informations */

   /* ==========================================  */
   /* Lets try to get common prefix. ===========  */
   /* ==========================================  */
   getparstring("prefix", &prefix);
   // if ( prefix == NULL )  Who cares?

   /* ==========================================  */
   /* Receiver points information ==============  */
   /* ==========================================  */
   getparstring("rps", &Rname);
   if ( Rname == NULL ) {
      if ( prefix == NULL )
         err("**** Missing required parameter 'rps=', see self-doc");
      //  I have a prefix and need build this file name.
      Rname = calloc(strlen(prefix)+2,sizeof(cwp_String));
      if ( Rname == NULL )
         err("**** Unable to allocate memory for R-File name.");
      strcpy(Rname,prefix);
      strcat(Rname,".r");
   }
   if (verbose)
      warn("Receiver file: %s",Rname);

   fpR = efopen(Rname, "r");
   if ( fpR == NULL )
         err("**** Error opening the receiver information file");

   numR = countRec(fpR, 'R');
   warn("Found %d receiver points entries.",numR);
   if ( numR == 0 )
      err("**** No receiver point information found!!");

   // Allocate area for the receiver informations
   // calloc zero out memory.
   RecInfo = calloc(numR,sizeof(struct PointInfo));
   if ( RecInfo == NULL ){
      err("**** Unable to allocate memory for Receiver Point Informations!!");
   }

   //  Get all Receiver Points info into memory
   RC = getPoints(fpR, 'R', rev, RecInfo);
   if ( RC != numR) {
      warn("Short read for Receiver Point Informations!");
      warn("Read %d receiver points entries instead of expected%d.",RC,numR);
   }
   if (verbose) {
      warn("Stored: %d Receiver Point Informations.",RC);
      for (i=0; i<RC; i++)
          warn("R %10.2f,%10.2f,%4.1f,%10.2lf,%10.2lf,%5.1f",
          RecInfo[i].Line,
          RecInfo[i].Point,
          RecInfo[i].PDepth,
          RecInfo[i].X,
          RecInfo[i].Y,
          RecInfo[i].Elev);
   }


   /* ==========================================  */
   /* Source points information ================  */
   /* ==========================================  */
   getparstring("sps", &Sname);
   if ( Sname == NULL ) {
      if ( prefix == NULL )
         err("**** Missing required parameter 'sps=', see self-doc");
      //  I have a prefix and need build this file name.
      Sname = calloc(strlen(prefix)+2,sizeof(cwp_String));
      if ( Sname == NULL )
         err("**** Unable to allocate memory for S-File name.");
      strcpy(Sname,prefix);
      strcat(Sname,".s");
   }
   if (verbose)
      warn("Source file: %s",Sname);

   fpS = efopen(Sname, "r");
   if ( fpS == NULL )
      err("**** Error opening the source information file");

   numS = countRec(fpS, 'S');
   warn("Found %d source points entries.",numS);
   if ( numS == 0 )
      err("**** No source point information found!!");

   // Allocate area for the Source informations
   // calloc zero out memory.
   SrcInfo = calloc(numS,sizeof(struct PointInfo));
   if ( SrcInfo == NULL ){
      err("**** Unable to allocate memory for Source Point Informations!!");
   }

   //  Get all Source Point info into memory
   RC = getPoints(fpS, 'S', rev, SrcInfo);
   if ( RC != numS) {
      warn("Short read for Source Point Informations!");
      warn("Read %d source point entries instead of expected%d.",RC,numS);
   }
   if (verbose) {
      warn("Stored: %d Source Point Informations.",RC);
      for (i=0; i<RC; i++)
          warn("S %10.2f,%10.2f,%4.1f,%10.2lf,%10.2lf,%5.1f",
          SrcInfo[i].Line,
          SrcInfo[i].Point,
          SrcInfo[i].PDepth,
          SrcInfo[i].X,
          SrcInfo[i].Y,
          SrcInfo[i].Elev);
   }
   /* ------------------------------------------  */

   /* ==========================================  */
   /* Relation (records) information ===========  */
   /* ==========================================  */
   getparstring("xps", &Xname);
   if ( Xname == NULL ) {
      if ( prefix == NULL )
         err("**** Missing required parameter 'xps=', see self-doc");
      //  I have a prefix and need build this file name.
      Xname = calloc(strlen(prefix)+2,sizeof(cwp_String));
      if ( Xname == NULL )
         err("**** Unable to allocate memory for X-File name.");
      strcpy(Xname,prefix);
      strcat(Xname,".x");
   }
   if (verbose)
      warn("Relation file: %s",Xname);

   fpX = efopen(Xname, "r");
   if ( fpX == NULL )
      err("**** Error opening the relation information file");

   numX = countRec(fpX, 'X');
   warn("Found %d relation information entries.",numX);
   if ( numX == 0 )
      err("**** No relation information found!!");

   // Allocate area for the Register informations
   // calloc zero out memory.
   FileInfo = calloc(numX,sizeof(struct RegInfo));
   if ( FileInfo == NULL ){
      err("**** Unable to allocate memory for Registers Informations!!");
   }

   //  Get all Record info into memory
   RC = getFiles(fpX, rev, FileInfo);
   if ( RC != numX) {
      warn("Short read for File Informations!");
      warn("Read %d file entries instead of expected%d.",RC,numX);
   }
   if (verbose) {
      warn("Stored: %d Register Information entries.",RC);
      for (i=0; i<RC; i++)
          warn("X %4d,%1d,%9.1f,%9.1f,%4d,%4d,%1d,%9.1f,%9.1f,%10.1f,%1d",
          FileInfo[i].Num,
          FileInfo[i].Inc,
          FileInfo[i].SLine,
          FileInfo[i].SPoint,
          FileInfo[i].FChan,
          FileInfo[i].TChan,
          FileInfo[i].IncChan,
          FileInfo[i].RLine,
          FileInfo[i].FRecv,
          FileInfo[i].TRecv,
          FileInfo[i].RInc);
   }
   /* ------------------------------------------  */
   /* ==========================================  */
   /* ==========================================  */
   /* ==========================================  */

   /* Get info from first trace */ 
   if (!gettr(&tr))  err("can't get first trace");
   seismic = ISSEISMIC(tr.trid);
   if (seismic) 
   {
      if (verbose)   warn("input is seismic data, trid=%d",tr.trid);
      // dt = ((double) tr.dt)/1000000.0; // Don't use dt
   }
   else 
   {
      if (verbose) warn("input is not seismic data, trid=%d",tr.trid);
      // dt = tr.d1; // Don't use dt
   }

   /* error trapping so that the user can have a default value of dt */
   if (!(dt || getparfloat("dt", &dt))) {
      // dt = .004;  // Don't use dt
      warn("WARNING:  neither dt nor d1 are set, nor is dt getparred!");
      warn("WARNING:  It is not used, just checked!!");
   }

   //nt = tr.ns;
   oldfldr  = 0;  //  fldr for last trace processed
   oldskip = 0;   //  still skipping the same fldr?
   curx = -1;      //  relation entry control
   curtrl = 0;    //  keeping tracl of current trace (in case of skip)

      
   /* ------------------------------------------  */

   /* ==========================================  */
   /* ======  Main loop over traces  ===========  */
   /* ==========================================  */

   do {

      if (tr.fldr != oldfldr) { //  Ok, we have a new fldr.
         // Advance relation entry?
         if 
            (curx+1 < numX && tr.fldr == FileInfo[curx+1].Num) {
            curx++;
            if ( curx == numX ) break;  //  No more relation available.
            // fprintf(stderr,"==> 1-curx %d\n",curx); // Debug
            // Process this one
            if ( verbose )
               warn("** processing fldr=%d",tr.fldr);
            oldfldr = tr.fldr;
            //fprintf(stderr,"==> New file %d: %d\n",curx,tr.fldr); // Debug
         }
         else {  // skip until the next file is found
            if ( tr.fldr != oldskip ) {  //  Skipping new fldr!
               if ( curx+1 == numX) {
                  curx++;
                  break;  //  No more relation available
               }
               warn("** skipping fldr=%d, looking for %d", 
                     tr.fldr,FileInfo[curx+1].Num);
               oldskip = tr.fldr;
            }
            continue;
         }
      }  // (tr.fldr != oldfldr)

      //  Ok, we are processing current fldr
      //  Is this channel within range?
      if ( tr.tracf < FileInfo[curx].FChan ) { // skip this one
         warn("** skipping fldr=%d, tracf=%d, looking for tracf=%d ",
               tr.fldr,tr.tracf,FileInfo[curx].FChan);
         continue;
      }
      else if ( tr.tracf > FileInfo[curx].TChan ) { // <FIXME> still unsure
              if ( curx+1 == numX) { // Is there any more relation info?
                 curx++; // Signal the end of table
                 break;  // Exit, no more relation available
              }
              if ( tr.fldr == FileInfo[curx+1].Num ) { // Is there more info?
                 do {
                    // Advance the register info
                    curx++;
                    if ( tr.tracf >= FileInfo[curx].FChan &&  // If chan ok,
                          tr.tracf <= FileInfo[curx].TChan) { // exit loop
                       break; // Found it
                    }

                 } while (tr.fldr == FileInfo[curx+1].Num);
                 //  Why did I get here?
                 //  do we have a new fldr?
                 if ( tr.tracf < FileInfo[curx].FChan ||
                      tr.tracf > FileInfo[curx].TChan) { // Ok, we have a new fldr.
                    //  skip this trace, look for a new file
                    continue;
                 }

              }
      }

      //  Ok, seems that we are good to process this one.
      //  Lets update the header
      //  We will keep fldr and trcf untoutched
      //  But will update tracl
      curtrl++;
      tr.tracl = curtrl;


      if ( tr.ep == 0 ) // Only if zeroed out, keep otherwise
         tr.ep = FileInfo[curx].SPoint; // No fractionary sp so far.

      tr.cdp = 0;  // not updating it yet

      //  This computes the receiver station 
      //  <FIXME>  Not sure if this use of IncChan is correct. :(
      //  It should group each IncChan channels in a single station
      //  for multicomponent data.  
      recsta = FileInfo[curx].FRecv + 
               (FileInfo[curx].TRecv-FileInfo[curx].FRecv) * 
               ((tr.tracf-FileInfo[curx].FChan)/FileInfo[curx].IncChan)/
               (FileInfo[curx].TChan-FileInfo[curx].FChan);

      //  Get the index of the point at corresponding point  structure
      idxR = GetPointIndex(RecInfo, numR, recsta,FileInfo[curx].RLine);
      idxS = GetPointIndex(SrcInfo, numS, FileInfo[curx].SPoint,
                           FileInfo[curx].SLine);
      // <FIXME> - Take care for information not available!
      // if ( idx[[R|S] < 0 ) { failed!  Skip trace/file! }

      // Avoid division by zero.
      if ( tr.scalel == 0 ) tr.scalel = 1;
      if ( tr.scalco == 0 ) tr.scalco = 1;
      
      // The scale factor
      scalel = 1./tr.scalel;
      scalco = 1./tr.scalco;
      if ( tr.scalel  < 0 )  scalel = -tr.scalel;
      if ( tr.scalco  < 0 )  scalco = -tr.scalco;

      //  Source coordinates
      tr.sx = SrcInfo[idxS].X * scalco;
      tr.sy = SrcInfo[idxS].Y * scalco;
      //  Receiver coordinates
      tr.gx = RecInfo[idxR].X * scalco;
      tr.gy = RecInfo[idxR].Y * scalco;

      if ( ibin != 0 )  {  // compute CDP number

         //  CDP Position (where to store?)
         Xcdp = (RecInfo[idxR].X+SrcInfo[idxS].X)/2.;
         Ycdp = (RecInfo[idxR].Y+SrcInfo[idxS].Y)/2.;

         //  First try for cdp computation.
         //  (may work for 2D straight lines)
         //
         temp = sqrt(pow(Xcdp-RecInfo[0].X,2) + 
                     pow(Ycdp-RecInfo[0].Y,2));
         tr.cdp = 100.5 + temp / ibin;  // 100 is just arbitrary
                                        // .5  if for rounding :P
         // Debug: print cdp# Xcdp Ycdp
         //fprintf(stderr,"cdp %5d  %10.2f  %10.2f\n",tr.cdp,Xcdp,Ycdp);

      }

      // Elevations
      tr.gelev = RecInfo[idxR].Elev * scalel;
      tr.selev = SrcInfo[idxS].Elev * scalel;
      tr.sdepth = SrcInfo[idxS].PDepth * scalel;

      // Offset - The offset signal is just an educated guess.  :/
      tr.offset = sqrt(pow(SrcInfo[idxS].X-RecInfo[idxR].X,2) + 
                       pow(SrcInfo[idxS].Y-RecInfo[idxR].Y,2));
      if ( SrcInfo[idxS].Point > RecInfo[idxR].Point )
         tr.offset = -tr.offset;

      // Assuming all coordinates in UTM projection
      tr.counit = 1;

      tr.sstat = SrcInfo[idxS].StaCor;
      tr.gstat = RecInfo[idxR].StaCor;

      puttr(&tr);
      nproct++; // count processed traces

   } while (gettr(&tr));

   /* ==========================================  */
   /* =========  End of Main loop ==============  */
   /* ==========================================  */

   /* ------------------------------------------  */

   if ( curx >= numX ) {
      warn("***  Last record processed -> fldr %d  ***",oldfldr);
      warn("***  Records after this were discarded ***");
   }

   warn("***  %d traces processed...",nproct);

   return(CWP_Exit());
}

int countRec(FILE *pfile, char type)
{
   //  This function read the file *pfile looking for lines with
   //  1st character == Type.   The file must be already oppened
   //  and will be repositioned at start point at exit.

   int tcount = 0;
   int count = 0;

   while (fgets(textbuffer, sizeof textbuffer, pfile) != NULL) /* read a line */
   {
       if (textbuffer[0] == type )
       {
          count++;
       }
       tcount++;
   }

   if (verbose)
      warn("Read %d entries, %d of type %c\n",tcount,count,type);

   /* reposition input file */
   fseek(pfile, 0L, SEEK_SET);

   return count;

}

int getPoints(FILE *pfile, char type, int rev, struct  PointInfo *Points)
{

   //  Struct PointInfo
   //  Line:    1- Line name
   //  Point:   2- Point number (station, PT, etc)
   //  PDepth:  6- Point depth in relation to surface (ex buried SP)
   //  X:      10- X coordinate
   //  Y:      11- Y coordinate
   //  Elev:   12- Surface elevation at point
   //
   //  Internal variables
   int count = 0;

   while (fgets(textbuffer, sizeof(textbuffer), pfile) != NULL) /*read a line*/
   {
       if (textbuffer[0] == type ) // Is it the desirable type?
       {
          // Parse record for desired fields

          getSPSfield(textbuffer, rev,  1, &Points[count].Line);  // 1
          getSPSfield(textbuffer, rev,  2, &Points[count].Point); // 2
          getSPSfield(textbuffer, rev,  5, &Points[count].StaCor);// 5
          getSPSfield(textbuffer, rev,  6, &Points[count].PDepth);// 6
          getSPSfield(textbuffer, rev, 10, &Points[count].X);     //10
          getSPSfield(textbuffer, rev, 11, &Points[count].Y);     //11
          getSPSfield(textbuffer, rev, 12, &Points[count].Elev);  //12

          count++;
       }
   }
   return count;

}

int getFiles(FILE*pfile,  int rev, struct  RegInfo* Files)
{

// Structure for relation (aka register) information

   // Num:        2- Field Record Number
   // Inc:        3- Field Record Increment
   // SLine:      5- Source Line
   // SPoint:     6- Source Point
   // FChan:      8- Start Channel
   // TChan:      9- To Channel
   // IncChan:   10- Channel Increment for multicomponent data
   // RLine:     11- Receiver Line
   // FRecv:     12- From Receiver Station
   // TRecv:     13- To receiver Station
   // RInc:      14- Receiver Index
   //
   //  Internal variables
   int count = 0;

   while (fgets(textbuffer, sizeof(textbuffer), pfile) != NULL) /*read a line*/
   {
       if (textbuffer[0] == 'X' ) // Is it the desirable type?
       {
          // Initialize a few for nonzero values
          Files[count].Inc = 1;
          Files[count].IncChan = 1;
          Files[count].RInc = 1;

          // Parse record for desired fields

          getSPSfield(textbuffer, rev,  2, &Files[count].Num);    
          getSPSfield(textbuffer, rev,  3, &Files[count].Inc);   
          getSPSfield(textbuffer, rev,  5, &Files[count].SLine);
          getSPSfield(textbuffer, rev,  6, &Files[count].SPoint);
          getSPSfield(textbuffer, rev,  8, &Files[count].FChan);
          getSPSfield(textbuffer, rev,  9, &Files[count].TChan);
          getSPSfield(textbuffer, rev, 10, &Files[count].IncChan); 
          getSPSfield(textbuffer, rev, 11, &Files[count].RLine);
          getSPSfield(textbuffer, rev, 12, &Files[count].FRecv);
          getSPSfield(textbuffer, rev, 13, &Files[count].TRecv);
          getSPSfield(textbuffer, rev, 14, &Files[count].RInc);

          count++;
       }
   }
   return count;

}


int GetPointIndex(struct PointInfo *Points,int nPoints, float Point,float Line)
{

   //  This function sweeps the PointInfo Points looking for an entry with the
   //  Point and Line.
   //  Upon finding it returns the index value.
   int i;

   for (i=0; i<nPoints; i++)
   {
       if ( abs(Points[i].Line  - Line ) <= 0.01 && // Float compair :0
            abs(Points[i].Point - Point) <= 0.01 )
          return i;
   }

   // failled!
   return -1;

}
