<script lang="ts">
  import { lcmLiveStatus, LCMLiveStatus, streamId } from '$lib/lcmLive';
  import { getPipelineValues } from '$lib/store';

  import Button from '$lib/components/Button.svelte';
  import Floppy from '$lib/icons/floppy.svelte';
  import { snapImage } from '$lib/utils';

  $: isLCMRunning = $lcmLiveStatus !== LCMLiveStatus.DISCONNECTED;
  $: console.log('isLCMRunning', isLCMRunning);
  let imageEl: HTMLImageElement;

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
</script>

<!-- CSS-only fullscreen overlay -->
<div id="fullscreen-overlay" class="fullscreen-overlay">
  {#if isLCMRunning && $streamId}
    <img
      bind:this={imageEl}
      class="fullscreen-image"
      src={'/api/stream/' + $streamId}
      alt="Fullscreen AI Image"
    />
  {/if}
  <a href="#" class="close-fullscreen">✕</a>
</div>

<div class="relative mx-auto aspect-square max-w-lg self-center overflow-hidden rounded-lg border border-slate-300">
  <!-- svelte-ignore a11y-missing-attribute -->
  {#if isLCMRunning && $streamId}
    <img
      class="aspect-square w-full rounded-lg"
      src={'/api/stream/' + $streamId}
    />
    <div class="absolute top-2 right-2">
      <a 
        href="#fullscreen-overlay"
        class="inline-block text-xl text-white p-3 shadow-lg rounded-lg opacity-75 hover:opacity-100 bg-black/50 no-underline"
        title="Fullscreen"
      >
        📺
      </a>
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

<style>
  .fullscreen-overlay {
    position: fixed;
    z-index: 99;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.95);
    display: flex;
    align-items: center;
    justify-content: center;
    visibility: hidden;
    opacity: 0;
    transition: opacity 0.3s ease;
  }

  .fullscreen-overlay:target {
    visibility: visible;
    opacity: 1;
  }

  .fullscreen-image {
    max-width: 95%;
    max-height: 95%;
    width: auto;
    height: auto;
    object-fit: contain;
    transform: scale(0.95);
    transition: transform 0.3s ease;
  }

  .fullscreen-overlay:target .fullscreen-image {
    transform: scale(1);
  }

  .close-fullscreen {
    position: absolute;
    top: 20px;
    right: 30px;
    color: white;
    font-size: 2rem;
    text-decoration: none;
    opacity: 0.8;
    transition: opacity 0.2s ease;
  }

  .close-fullscreen:hover {
    opacity: 1;
  }
</style>
