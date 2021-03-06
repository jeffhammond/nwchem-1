      SUBROUTINE OFFSET_ccsdtq_lr_alpha_3_1_1(l_a_offset,k_a_offset,size
     &)
C     $Id$
C     This is a Fortran77 program generated by Tensor Contraction Engine v.1.0
C     Copyright (c) Battelle & Pacific Northwest National Laboratory (2002)
C     i2 ( p8 p5 )_ob
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
      INTEGER p8b
      INTEGER p5b
      length = 0
      DO p8b = noab+1,noab+nvab
      DO p5b = noab+1,noab+nvab
      IF (int_mb(k_spin+p8b-1) .eq. int_mb(k_spin+p5b-1)) THEN
      IF (ieor(int_mb(k_sym+p8b-1),int_mb(k_sym+p5b-1)) .eq. irrep_ob) T
     &HEN
      IF ((.not.restricted).or.(int_mb(k_spin+p8b-1)+int_mb(k_spin+p5b-1
     &).ne.4)) THEN
      length = length + 1
      END IF
      END IF
      END IF
      END DO
      END DO
      IF (.not.MA_PUSH_GET(mt_int,2*length+1,'noname',l_a_offset,k_a_off
     &set)) CALL ERRQUIT('ccsdtq_lr_alpha_3_1_1',0,MA_ERR)
      int_mb(k_a_offset) = length
      addr = 0
      size = 0
      DO p8b = noab+1,noab+nvab
      DO p5b = noab+1,noab+nvab
      IF (int_mb(k_spin+p8b-1) .eq. int_mb(k_spin+p5b-1)) THEN
      IF (ieor(int_mb(k_sym+p8b-1),int_mb(k_sym+p5b-1)) .eq. irrep_ob) T
     &HEN
      IF ((.not.restricted).or.(int_mb(k_spin+p8b-1)+int_mb(k_spin+p5b-1
     &).ne.4)) THEN
      addr = addr + 1
      int_mb(k_a_offset+addr) = p5b - noab - 1 + nvab * (p8b - noab - 1)
      int_mb(k_a_offset+length+addr) = size
      size = size + int_mb(k_range+p8b-1) * int_mb(k_range+p5b-1)
      END IF
      END IF
      END IF
      END DO
      END DO
      RETURN
      END
