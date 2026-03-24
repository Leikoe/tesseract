/**
 * Tests for flat-mode for loops (looping edges in the main graph).
 *
 * The `for` node's iter_var/current_i outputs define the loop body.
 * Body nodes feed back into update_i inputs (looping edges).
 */

import { describe, it, expect } from 'vitest';
import { tile, ptr } from '../../src/types.js';
import { emitMLIR } from '../../src/codegen.js';
import { GraphBuilder, makePtrChain } from '../helpers.js';

describe('flat for-loop codegen', () => {
  it('emits a loop with a carried accumulator', () => {
    const b = new GraphBuilder('sum_loop', 'kernel');

    const ptrType = tile([], ptr('f32'));
    const aPtr = b.entryArg('a_ptr', ptrType);

    // Loop bounds: for i in 0..10 step 1
    const zero = b.constant('0', 'i32');
    const ten = b.constant('10', 'i32');
    const one = b.constant('1', 'i32');
    const initVal = b.constant('0.0', 'f32', [128]);

    // Create flat for node
    const loop = b.forCreate(zero, ten, one, [{ name: 'sum', init: initVal }], 'i');

    // Body: offset pointers by iteration variable, load, add to accumulator
    const offsets = b.iota([128], 'i32');
    const aPtrs = makePtrChain(b, aPtr, offsets, 128);

    const ivReshaped = b.reshape(loop.iterVar, [1]);
    const ivBroadcast = b.broadcast(ivReshaped, [128]);
    const shiftedPtrs = b.offset(aPtrs, ivBroadcast);
    const { data: loaded } = b.loadPtr(shiftedPtrs, 'weak');

    const newSum = b.addf(loop.current[0], loaded);

    // Looping edge: body result → for.update_0
    b.connectForUpdate(loop.nodeId, 0, newSum);

    // After loop: store final result
    const outPtrs = makePtrChain(b, aPtr, offsets, 128);
    b.storePtr(outPtrs, loop.result[0], 'weak');

    const mlir = emitMLIR(b.build());

    expect(mlir).toContain('for %i in');
    expect(mlir).toContain('iter_values(%sum =');
    expect(mlir).toContain('continue');
    expect(mlir).toContain('addf');
    expect(mlir).toContain('load_ptr_tko');

    expect(mlir).toBe(
      [
        'cuda_tile.module @sum_loop {',
        '    entry @kernel(%a_ptr : tile<ptr<f32>>) {',
        '        %0 = constant <i32: 0> : tile<i32>',
        '        %1 = iota : tile<128xi32>',
        '        %2 = constant <i32: 10> : tile<i32>',
        '        %3 = constant <i32: 1> : tile<i32>',
        '        %4 = constant <f32: 0.0> : tile<128xf32>',
        '        %5 = reshape %a_ptr : tile<ptr<f32>> -> tile<1xptr<f32>>',
        '        %6 = reshape %a_ptr : tile<ptr<f32>> -> tile<1xptr<f32>>',
        '        %7 = broadcast %5 : tile<1xptr<f32>> -> tile<128xptr<f32>>',
        '        %8 = broadcast %6 : tile<1xptr<f32>> -> tile<128xptr<f32>>',
        '        %9 = offset %7, %1 : tile<128xptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>',
        '        %10 = offset %8, %1 : tile<128xptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>',
        '        %11 = for %i in (%0 to %2, step %3) : tile<i32> iter_values(%sum = %4) -> (tile<128xf32>) {',
        '            %12 = reshape %i : tile<i32> -> tile<1xi32>',
        '            %13 = broadcast %12 : tile<1xi32> -> tile<128xi32>',
        '            %14 = offset %9, %13 : tile<128xptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>',
        '            %15, %16 = load_ptr_tko weak %14 : tile<128xptr<f32>> -> tile<128xf32>, token',
        '            %17 = addf %sum, %15 rounding<nearest_even> : tile<128xf32>',
        '            continue %17 : tile<128xf32>',
        '        }',
        '        %18 = store_ptr_tko weak %10, %11 : tile<128xptr<f32>>, tile<128xf32> -> token',
        '        return',
        '    }',
        '}',
      ].join('\n'),
    );
  });
});
