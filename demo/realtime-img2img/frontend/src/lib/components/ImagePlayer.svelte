<script lang="ts">
  import { lcmLiveStatus, LCMLiveStatus, streamId } from '$lib/lcmLive';
  import { getPipelineValues } from '$lib/store';

  import Button from '$lib/components/Button.svelte';
  import Floppy from '$lib/icons/floppy.svelte';
  import { snapImage } from '$lib/utils';

  $: isLCMRunning = $lcmLiveStatus !== LCMLiveStatus.DISCONNECTED;
  $: console.log('isLCMRunning', isLCMRunning);
  let imageEl: HTMLImageElement;
  let isFullscreen = false;

  async function takeSnapshot() {
    if (isLCMRunning) {
      await snapImage(imageEl, {
        prompt: getPipelineValues()?.prompt,
        negative_prompt: getPipelineValues()?.negative_prompt,
        seed: getPipelineValues()?.seed,
        guidance_scale: getPipelineValues()?.guidance_scale
      });
    }
  }

  function toggleFullscreen() {
    isFullscreen = !isFullscreen;
  }

  function handleKeydown(event: KeyboardEvent) {
    if (event.key === 'Escape' && isFullscreen) {
      isFullscreen = false;
    }
  }
</script>

<svelte:window on:keydown={handleKeydown} />

<div
  class="relative mx-auto aspect-square max-w-lg self-center overflow-hidden rounded-lg border border-slate-300"
  class:hidden={isFullscreen}
>
  <!-- svelte-ignore a11y-missing-attribute -->
  {#if isLCMRunning && $streamId}
    <img
      bind:this={imageEl}
      class="aspect-square w-full rounded-lg"
      src={'/api/stream/' + $streamId}
    />
    <div class="absolute top-2 right-2">
      <Button
        on:click={toggleFullscreen}
        title={'Fullscreen'}
        classList={'text-xl text-white p-3 shadow-lg rounded-lg opacity-75 hover:opacity-100 bg-black/50'}
      >
        📺
      </Button>
    </div>
    <div class="absolute bottom-1 right-1">
      <Button
        on:click={takeSnapshot}
        disabled={!isLCMRunning}
        title={'Take Snapshot'}
        classList={'text-sm ml-auto text-white p-1 shadow-lg rounded-lg opacity-50'}
      >
        <Floppy classList={''} />
      </Button>
    </div>
  {:else}
    <img
      class="aspect-square w-full rounded-lg"
      src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    />
  {/if}
</div>

<!-- Fullscreen overlay -->
{#if isFullscreen && isLCMRunning && $streamId}
  <div 
    class="fixed inset-0 z-50 bg-black flex items-center justify-center"
    on:click={toggleFullscreen}
    role="button"
    tabindex="0"
    on:keydown={(e) => e.key === 'Enter' && toggleFullscreen()}
  >
    <img
      bind:this={imageEl}
      class="max-h-full max-w-full object-contain will-change-auto"
      src={'/api/stream/' + $streamId}
      alt="Fullscreen AI Image"
      loading="eager"
    />
    <button
      class="absolute top-4 right-4 text-white text-2xl hover:text-gray-300 transition-colors duration-200"
      on:click|stopPropagation={toggleFullscreen}
    >
      ✕
    </button>
  </div>
{/if}
