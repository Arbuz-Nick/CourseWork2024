<template>
  <q-item
    :class="[$style.actionItem, $style.button]"
    clickable
    dense
    tag="a"
    target="_blank"
    @click="requestFile"
  >
    <q-item-section>
      <q-item-label :class="$style.itemLabel">Upload file</q-item-label>
    </q-item-section>
  </q-item>

  <q-item
    :class="[$style.actionItem, $style.button]"
    clickable
    dense
    tag="a"
    target="_blank"
    @click="denoiseFile"
    v-if="noisedBasename !== undefined"

  >
    <q-item-section>
      <q-item-label :class="$style.itemLabel">Run</q-item-label>
    </q-item-section>
  </q-item>

  <q-item
    :class="[$style.actionItem, $style.button]"
    clickable
    dense
    tag="a"
    target="_blank"
    :href="denoisedItem?.path"
    v-if="denoisedItem !== undefined"
  >
    <q-item-section>
      <q-item-label :class="$style.itemLabel"
        >Download {{ denoisedItem?.basename }}</q-item-label
      >
    </q-item-section>
  </q-item>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue';
declare const window: any;

const denoisedItem = ref<{ basename: string; path: string } | undefined>(
  undefined
);
const noisedBasename = ref<string | undefined>(undefined);

function requestFile() {
  console.warn('try to request');
  denoisedItem.value = undefined;
  noisedBasename.value = undefined;
  noisedBasename.value = window.bridge.requestFile();
}

async function denoiseFile() {
  denoisedItem.value = await window.bridge.denoiseFile();
}

onMounted(() => {});
</script>

<style module lang="scss">
$border-radius: 6px;
$padding: 0px 4px;
$gap: 8px;
$icon-height: 60%;

.actionItem {
  border: none;
  cursor: pointer;
  border-radius: $border-radius;
  background: white;
  padding: $padding;
  display: flex;
  flex-direction: row;
  gap: $gap;
  margin: 0;
  justify-self: center;
  align-items: center;
  align-content: center;
  margin: 10px;

  & > :global(.q-focus-helper) {
    display: none !important;
    visibility: hidden !important;
  }

  &:hover {
    background: lightgray;
  }

  .itemLabel {
    text-overflow: ellipsis;
    overflow: hidden;
    white-space: nowrap;
  }

  .itemIconSection {
    padding: 0;
    margin: 0;
    width: fit-content;
    max-width: fit-content;
    min-width: fit-content;

    .itemIcon {
      height: $icon-height;
      opacity: 0.5;
    }
  }
  &.selectedItem {
    background: lightblue;
  }

  &.button {
    .itemLabel {
      text-align: center;
    }
  }

  &.listItem {
    width: 100%;
    min-width: 100%;
    max-width: 100%;
  }
}
</style>
