      SUBROUTINE tce_dens_hh_offset(l_a_offset,k_a_offset,size)
c     written by Peng-Dong Fan, 2007/07/24
C     dens_hh ( h5 h6 )
      IMPLICIT NONE
#include "global.fh"
#include "mafdecls.fh"
#include "sym.fh"
#include "errquit.fh"
#include "tce.fh"
      INTEGER l_a_offset
      INTEGER k_a_offset
      INTEGER size
      INTEGER length
      INTEGER addr
      INTEGER h5b
      INTEGER h6b
      length = 0
      DO h5b = 1,noab
      DO h6b = 1,noab
      IF (int_mb(k_spin+h5b-1) .eq. int_mb(k_spin+h6b-1)) THEN
      IF (ieor(int_mb(k_sym+h5b-1),int_mb(k_sym+h6b-1)) .eq. irrep_t) TH
     &EN
      IF ((.not.restricted).or.(int_mb(k_spin+h5b-1)+int_mb(k_spin+h6b-1
     &).ne.4)) THEN
      length = length + 1
      END IF
      END IF
      END IF
      END DO
      END DO
      IF (.not.MA_PUSH_GET(mt_int,2*length+1,'denshh',l_a_offset,k_a_off
     &set)) CALL ERRQUIT('tce_dens_hh_offset',0,MA_ERR)
      int_mb(k_a_offset) = length
      addr = 0
      size = 0
      DO h5b = 1,noab
      DO h6b = 1,noab
      IF (int_mb(k_spin+h5b-1) .eq. int_mb(k_spin+h6b-1)) THEN
      IF (ieor(int_mb(k_sym+h5b-1),int_mb(k_sym+h6b-1)) .eq. irrep_t) TH
     &EN
      IF ((.not.restricted).or.(int_mb(k_spin+h5b-1)+int_mb(k_spin+h6b-1
     &).ne.4)) THEN
      addr = addr + 1
      int_mb(k_a_offset+addr) = h6b - 1 + noab * (h5b - 1)
      int_mb(k_a_offset+length+addr) = size
      size = size + int_mb(k_range+h5b-1) * int_mb(k_range+h6b-1)
      END IF
      END IF
      END IF
      END DO
      END DO
      RETURN
      END
c $Id$
